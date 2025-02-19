import torch
import wandb
from torch import nn

from beam_search import beam_search_batched, beam_search_unbatched
from constants import constants
from data_processing.dataloader import get_data_loader
from data_processing.vocab import load_vocab, output_to_text
from hyperparameters import hyperparameters
from models.transformer_model import TransformerModel
from utils.checkpoints import load_checkpoint
from uq.acquisition_func import BeamScore, BLEUVariance
from validate import validate

# RESULTS
# - embedding_fix:
#    - Greedy BLEU: 22.24
#    - Beam BLEU: 22.42

def main() -> None:
    # Load shared vocabulary
    # wandb.restore("checkpoints/checkpoint-175000.pth", run_path="sondresorbye-magson/TransformerUQ/54inz442")  # type: ignore
    src_vocab = load_vocab(constants.file_paths.src_vocab)
    tgt_vocab = load_vocab(constants.file_paths.tgt_vocab)
    model: nn.Module = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    ).to(hyperparameters.device)

    if torch.cuda.is_available():
        model = torch.compile(model)  # type: ignore
        torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load the checkpoint
    load_checkpoint(
        model, 
        optimizer, 
        "checkpoints/checkpoint-175000.pth",
        remove_orig_prefix=not torch.cuda.is_available()
    )

    # Set up the test data loader with the shared vocabulary
    test_loader = get_data_loader(
        src_file=constants.file_paths.bpe_test_de,
        tgt_file=constants.file_paths.bpe_test_en,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=32, # Needs to be low due to beam search
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    # Validate the model and calculate BLEU score
    bleu = validate(model, test_loader)
    print(f"BLEU Score on test_set: {bleu}")

if __name__ == "__main__":
    main()
