from hyperparameters import Hyperparameter
from vocab import PAD_TOKEN, load_vocab
from dataloader import get_data_loader
from models.transformer import Transformer
import torch
import torch.nn as nn
from validate import validate

def main():
    de_vocab = load_vocab("local/vocab_de.pkl")
    en_vocab = load_vocab("local/vocab_en.pkl")
    hyperparameters = Hyperparameter()

    test_ood_loader = get_data_loader(
        src_file="local/data/test_ood/bpe_test_ood.nl",
        tgt_file="local/data/test_ood/bpe_test_ood.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.max_len,
    )

    # Load model model weights from checkpoint
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
    )

    model.to(device)

    checkpoint_to_load = 2000
    model.load_state_dict(torch.load(f"checkpoints/checkpoint-{checkpoint_to_load}.pth")["model_state_dict"])

    # Validate model on OOD data
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.token_to_id(PAD_TOKEN))
    validate(model, test_ood_loader, criterion)


if __name__ == "__main__":
    main()