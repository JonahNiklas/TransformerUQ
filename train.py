import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fairseq.data import Dictionary, data_utils, LanguagePairDataset
from fairseq.models.transformer import TransformerModel
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

import validate

def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, max_updates: int, validate_every: int = 1000):
    model.train()
    for update in tqdm(range(max_updates), desc=f"Step {update}/{max_updates}"):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            src_tokens, src_lengths, tgt_tokens = batch['net_input']['src_tokens'], batch['net_input']['src_lengths'], batch['target']
            device = model.device
            src_tokens, src_lengths, tgt_tokens = src_tokens.to(device), src_lengths.to(device), tgt_tokens.to(device)

            optimizer.zero_grad()
            output = model(src_tokens, src_lengths, tgt_tokens)
            loss = criterion(output, tgt_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % validate_every == 0:
                validate(model, test_data)

    avg_loss = total_loss / max_updates
    print(f"Average Loss: {avg_loss}")

if __name__ == "__main__":
    # Load data
    data_dir = 'local/data-bin/iwslt14.tokenized.de-en'
    src_dict = Dictionary.load(f'{data_dir}/dict.de.txt')
    tgt_dict = Dictionary.load(f'{data_dir}/dict.en.txt')
    dataset = LanguagePairDataset(
        src=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.de', src_dict),
        src_sizes=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.de', src_dict).sizes,
        src_dict=src_dict,
        tgt=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.en', tgt_dict),
        tgt_sizes=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.en', tgt_dict).sizes,
        tgt_dict=tgt_dict,
    )
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    # Model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel.from_pretrained(data_dir, checkpoint_file='checkpoint_best.pt').to(device)
    criterion = LabelSmoothedCrossEntropyCriterion(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98))

    # Train
    train(model, dataloader, optimizer, criterion, device, max_updates=500_000)