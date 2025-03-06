from __future__ import annotations
from typing import Tuple
import tiktoken
import numpy as np
import torch
import math
import time
import os
import torch.nn.functional as F
import wandb

from gpt2project.dataloader import DataLoaderLite
from gpt2project.gpt2_hellaswag import (
    get_most_likely_row,
    iterate_examples,
    render_example,
)
from gpt2project.gpt2model import GPT, GPTConfig
from gpt2project.hyperparameters import hyperparameters
import logging

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from gpt2project.ddp import (
    ddp,
    ddp_rank,
    ddp_world_size,
    master_process,
    device,
    device_type,
    ddp_local_rank,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

torch.manual_seed(hyperparameters.generation.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(hyperparameters.generation.seed)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = hyperparameters.training.total_batch_size
B = hyperparameters.training.micro_batch_size
T = hyperparameters.training.sequence_length
max_steps = hyperparameters.training.max_steps
warmup_steps = hyperparameters.training.warmup_steps

assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    logger.info(f"total desired batch size: {total_batch_size}")
    logger.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")


def main() -> None:

    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
    )

    # Initialize wandb for experiment tracking
    if master_process:
        wandb.init(
            project=hyperparameters.wandb.project,
            entity=hyperparameters.wandb.entity,
            config=hyperparameters.model_dump(),
            dir=hyperparameters.wandb.dir,
            mode=hyperparameters.wandb.mode,
        )

    torch.set_float32_matmul_precision("high")

    # create model
    model = GPT(
        GPTConfig(
            vocab_size=hyperparameters.model.vocab_size,
            block_size=hyperparameters.model.block_size,
            n_layer=hyperparameters.model.n_layer,
            n_head=hyperparameters.model.n_head,
            n_embd=hyperparameters.model.n_embd,
        )
    )
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = (
        False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    )
    if use_compile:
        model = torch.compile(model)  # type: ignore
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # type: ignore
    raw_model = (
        model.module if ddp else model
    )  # always contains the "raw" unwrapped model

    max_lr = hyperparameters.training.max_lr
    min_lr = hyperparameters.training.min_lr

    def get_lr(it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimize!
    optimizer = raw_model.configure_optimizers(
        weight_decay=hyperparameters.training.weight_decay,
        learning_rate=hyperparameters.training.max_lr,
        device_type=device_type,
    )

    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # once in a while evaluate our validation loss
        if step % hyperparameters.training.evaluate_every == 0 or last_step:
            evaluate_validation_loss(
                model,
                val_loader,
                device,
                device_type,
                ddp,
                master_process,
                log_file,
                step,
                last_step,
            )

        # once in a while evaluate hellaswag
        if (step % hyperparameters.training.evaluate_every == 0 or last_step) and (
            not use_compile
        ):
            evaluate_hellaswag(
                model,
                device,
                device_type,
                ddp,
                ddp_rank,
                ddp_world_size,
                master_process,
                log_file,
                step,
            )

        # once in a while generate from the model (except step 0, which is noise)
        if (
            (step > 0 and step % hyperparameters.training.evaluate_every == 0)
            or last_step
        ) and (not use_compile):
            generate_from_model(
                model,
                device,
                device_type,
                ddp_rank,
                ddp_world_size,
                enc,
                step,
            )

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # type: ignore
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), hyperparameters.training.grad_clip
        )
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            logger.info(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

            # Log metrics to wandb
            if hyperparameters.wandb.enabled:
                wandb.log(
                    {
                        "train/loss": loss_accum.item(),
                        "train/learning_rate": lr,
                        "train/grad_norm": norm,
                        "train/tokens_per_sec": tokens_per_sec,
                    },
                    step=step,
                )

    if ddp:
        destroy_process_group()


def evaluate_validation_loss(
    model: GPT,
    val_loader: DataLoaderLite,
    device: str,
    device_type: str,
    ddp: bool,
    master_process: bool,
    log_file: str,
    step: int,
    last_step: bool,
) -> None:
    raw_model = model.module if ddp else model
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = torch.tensor(0.0, device=device)
        val_loss_steps = hyperparameters.training.val_loss_steps
        for val_loss_step_idx in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        logger.info(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # Log validation metrics to wandb
        if hyperparameters.wandb.enabled:
            wandb.log(
                {
                    "val/loss": val_loss_accum.item(),
                },
                step=step,
            )

        if step > 0 and (step % hyperparameters.training.save_every == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "val_loss": val_loss_accum.item(),
            }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)

            # Log model checkpoint as wandb artifact
            if hyperparameters.wandb.enabled:
                # Create a model artifact
                model_artifact = wandb.Artifact(
                    name=f"model-checkpoint-{step}",
                    type="model",
                    description=f"GPT-2 model checkpoint at step {step}",
                    metadata={
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "model_config": {
                            "vocab_size": raw_model.config.vocab_size,
                            "block_size": raw_model.config.block_size,
                            "n_layer": raw_model.config.n_layer,
                            "n_head": raw_model.config.n_head,
                            "n_embd": raw_model.config.n_embd,
                        },
                    },
                )

                # Add the model file to the artifact
                model_artifact.add_file(checkpoint_path)

                # Log the artifact to wandb
                wandb.log_artifact(model_artifact)


def evaluate_hellaswag(
    model: GPT,
    device: str,
    device_type: str,
    ddp: bool,
    ddp_rank: int,
    ddp_world_size: int,
    master_process: bool,
    log_file: str,
    step: int,
) -> None:
    model.eval()
    num_correct_norm: int | float | torch.Tensor = 0
    num_total: int | float | torch.Tensor = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(
            num_correct_norm, dtype=torch.long, device=device
        )
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        logger.info(
            f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")

        # Log hellaswag metrics to wandb
        if hyperparameters.wandb.enabled:
            wandb.log(
                {
                    "eval/hellaswag_accuracy": acc_norm,
                    "eval/hellaswag_correct": num_correct_norm,
                    "eval/hellaswag_total": num_total,
                },
                step=step,
            )


def generate_from_model(
    model: GPT,
    device: str,
    device_type: str,
    ddp_rank: int,
    ddp_world_size: int,
    enc: tiktoken.Encoding,
    step: int = 0,
) -> None:
    model.eval()
    num_return_sequences = hyperparameters.generation.num_return_sequences
    max_length = hyperparameters.generation.max_length
    tokens = torch.tensor(enc.encode("Hello, I'm a language model,"), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(hyperparameters.generation.seed + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    generated_samples = []
    for i in range(num_return_sequences):
        token_list = xgen[i, :max_length].tolist()
        decoded = enc.decode(token_list)
        logger.info(f"rank {ddp_rank} sample {i}: {decoded}")
        generated_samples.append(decoded)

    # Log generated text samples to wandb
    if ddp_rank == 0:  # Only log from the first process to avoid duplicates
        if hyperparameters.wandb.enabled:
            # Create a wandb Table for the generated samples
            columns = ["sample_id", "generated_text"]
            data = [[i, sample] for i, sample in enumerate(generated_samples)]
            table = wandb.Table(columns=columns, data=data)  # type: ignore
            wandb.log({"generated_samples": table}, step=step)


if __name__ == "__main__":
    main()
