import einops
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

def contrastive_loss(feats, t=0.07):
    feats = F.normalize(feats, dim=2)  # B x K x C
    scores = torch.einsum('aid, bjd -> abij', feats, feats)
    scores = einops.rearrange(scores, 'a b i j -> (a i) (b j)')

    # positive logits: Nx1
    pos_idx = einops.repeat(torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    pos_idx.fill_diagonal_(0)
    l_pos = torch.gather(scores, 1, pos_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # rand_idx = torch.randint(1, l_pos.size(1), (l_pos.size(0), 1), device=feats.device)
    l_pos, _ = l_pos.min(dim=1) #torch.gather(l_pos, 1, rand_idx)

    # negative logits: NxK
    neg_idx = einops.repeat(1-torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    l_neg = torch.gather(scores, 1, neg_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # logits: Nx(1+K)
    logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

    # apply temperature
    logits /= t

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, labels)
