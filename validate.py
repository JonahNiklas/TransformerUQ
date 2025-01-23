import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
from sacrebleu import corpus_bleu

def validate(model: nn.Module, test_data: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for i, batch in enumerate(test_data):
            src_tokens, src_lengths, tgt_tokens = batch['net_input']['src_tokens'], batch['net_input']['src_lengths'], batch['target']
            device = model.device
            src_tokens, src_lengths, tgt_tokens = src_tokens.to(device), src_lengths.to(device), tgt_tokens.to(device)

            output = model(src_tokens, src_lengths, tgt_tokens)
            loss = criterion(output, tgt_tokens)
            total_loss += loss.item()

            # Convert output to text
            hypotheses = model.decode(output)
            references = model.decode(tgt_tokens)

            all_hypotheses.extend(hypotheses)
            all_references.extend(references)

    avg_loss = total_loss / len(test_data)
    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score

    print(f"Validation Loss: {avg_loss}")
    print(f"BLEU Score: {bleu_score}")

    return bleu_score