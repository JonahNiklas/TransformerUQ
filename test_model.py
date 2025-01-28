import torch
import wandb
from dataloader import get_data_loader
from hyperparameters import hyperparameters
from models.transformer import Transformer
from train import load_checkpoint
from validate import validate
from vocab import load_vocab

def main():
    # Load model from wandb
    wandb.restore("checkpoints/checkpoint-500000.pth", run_path="sondresorbye-magson/TransformerUQ/5k0r04m7")
    en_vocab = load_vocab("local/vocab_en.pkl")
    de_vocab = load_vocab("local/vocab_de.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=hyperparameters.encoder_embed_dim,
        num_heads=hyperparameters.encoder_attention_heads,
        d_ff=hyperparameters.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.encoder_layers,
        num_decoder_layers=hyperparameters.encoder_layers,
        dropout=hyperparameters.dropout,
        max_len=hyperparameters.max_len,
    ).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    load_checkpoint(model, optimizer, "checkpoints/checkpoint-500000.pth")

    max_len=512
    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=max_len,
    )

    validate(model, test_loader, None)


if __name__ == '__main__':
    main()