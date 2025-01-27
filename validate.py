import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
from sacrebleu import corpus_bleu
from tqdm import tqdm
from generate import generate_autoregressivly

from vocab import output_to_text
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model: nn.Module, test_data: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []

    logger.debug("Started validating models")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_data), desc="Running validation", total=len(test_data)):
            src_tokens, tgt_tokens = batch
            src_tokens, tgt_tokens = src_tokens.to(device), tgt_tokens.to(device)
            
            output = generate_autoregressivly(model, src_tokens, print_ex=1)

            # Convert output to text
            hypotheses = output.argmax(dim=1)
            hypotheses = [output_to_text(hyp) for hyp in hypotheses.tolist()]
            references = [output_to_text(ref) for ref in tgt_tokens.tolist()]

            all_hypotheses.extend(hypotheses)
            all_references.extend(references)


    avg_loss = total_loss / len(test_data)
    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score

    print(f"Validation Loss: {avg_loss} | BLEU Score: {bleu_score}")

    return bleu_score

if __name__ == "__main__":
    dummy_hyptheses = [
        "<pos> This is goof the the the the",
    ]
    dummy_references = [
        "This is a test sentence for BLEU score calculation",
        "Banana is good for a long life",
        "cat is meowing",
        "house is big",
    ]

    bleu_score = corpus_bleu(dummy_hyptheses, [dummy_references]).score
    print(f"Dummy BLEU Score: {bleu_score}")