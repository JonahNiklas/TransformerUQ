import torch

def collate_fn(batch, pad_id=0):
    # batch: list of (src_ids, tgt_ids)
    src_batch, tgt_batch = zip(*batch)

    max_src_len = max(len(x) for x in src_batch)
    max_tgt_len = max(len(x) for x in tgt_batch)

    padded_src = []
    padded_tgt = []
    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        padded_src.append(src_ids + [pad_id]*(max_src_len - len(src_ids)))
        padded_tgt.append(tgt_ids + [pad_id]*(max_tgt_len - len(tgt_ids)))

    src_tensor = torch.tensor(padded_src, dtype=torch.long)
    tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)
    return src_tensor, tgt_tensor
