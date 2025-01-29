import torch
from torch import nn

import wandb
from dataloader import get_data_loader
from hyperparameters import hyperparameters
from models.transformer import Transformer
from train import load_checkpoint
from validate import validate
from vocab import load_vocab


def main() -> None:
    # Load model from wandb
    wandb.restore("checkpoints/checkpoint-500000.pth", run_path="sondresorbye-magson/TransformerUQ/5k0r04m7")  # type: ignore
    en_vocab = load_vocab("local/vocab_en.pkl")
    de_vocab = load_vocab("local/vocab_de.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = Transformer(
        vocab_size=len(de_vocab),
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    ).to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)  # type: ignore
        torch.set_float32_matmul_precision('high')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    load_checkpoint(model, optimizer, "checkpoints/checkpoint-500000.pth",remove_orig_prefix= not torch.cuda.is_available())

    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    bleu = validate(model, test_loader, None)
    print(f"BLEU Score: {bleu}")


if __name__ == "__main__":
    main()
