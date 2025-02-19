from EnDeTransformer.constants import constants
from EnDeTransformer.hyperparameters import hyperparameters
from EnDeTransformer.models.transformer_model import TransformerModel
from EnDeTransformer.data_processing.vocab import PAD_TOKEN, load_vocab
from EnDeTransformer.data_processing.dataloader import get_data_loader
from EnDeTransformer.models.transformer import Transformer
import torch
import torch.nn as nn
from EnDeTransformer.validate import validate

def main() -> None:
    vocab = load_vocab(constants.file_paths.vocab)

    test_ood_loader = get_data_loader(
        src_file="EnDeTransformer/local/data/test_ood/bpe_test_ood.nl",
        tgt_file="EnDeTransformer/local/data/test_ood/bpe_test_ood.en",
        vocab=vocab,
        batch_size=hyperparameters.training.batch_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    # Load model model weights from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = TransformerModel(
        vocab_size=len(vocab),
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    ).to(device)

    model.to(device)

    checkpoint_to_load = 175000
    model.load_state_dict(torch.load(f"EnDeTransformer/checkpoints/checkpoint-{checkpoint_to_load}.pth")["model_state_dict"])

    # Validate model on OOD data
    validate(model, test_ood_loader)


if __name__ == "__main__":
    main()