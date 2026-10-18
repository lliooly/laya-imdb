"""Task-conditioned dense/sparse document readout, independent of data access."""
import math

import torch
from torch import nn


KINDS = ('mean', 'dense', 'top32')


class TokenQueryHead(nn.Module):
    def __init__(self, kind, hidden_size=1024, width=128, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown pooling rule')
        self.kind, self.width = kind, width
        self.input_norm = nn.LayerNorm(hidden_size)
        self.tokens = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU())
        self.marker_projection = nn.Linear(hidden_size, width)
        self.queries = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.queries, std=.02)
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(width, 1)

    def forward(self, hidden, mask, markers, return_attention=False):
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or markers.shape != (len(hidden), 2):
            raise ValueError('Expected B x L x D, B x L mask and B x 2 markers')
        if not mask.any(-1).all() or (markers < 0).any() or (markers >= hidden.shape[1]).any():
            raise ValueError('Empty input or invalid marker')
        if not torch.gather(mask, 1, markers).all():
            raise ValueError('Marker must be an actual token')
        h = self.input_norm(hidden)
        v = self.tokens(h)
        marker_h = torch.gather(h, 1, markers[:, :, None].expand(-1, -1, h.shape[-1]))
        q = self.marker_projection(marker_h) + self.queries[None]
        if self.kind == 'mean':
            weights = (mask.float()/mask.sum(-1, keepdim=True))[:, None].expand(-1, 2, -1)
        else:
            scores = torch.einsum('bqd,bld->bql', q, v)/math.sqrt(self.width)
            scores = scores.masked_fill(~mask[:, None], -torch.inf)
            if self.kind == 'top32':
                indices = scores.topk(min(32, scores.shape[-1]), dim=-1).indices
                selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
                scores = scores.masked_fill(~selected, -torch.inf)
            weights = scores.softmax(-1)
        pooled = torch.einsum('bql,bld->bqd', weights, v)
        logits = self.scorer(self.dropout(self.output_norm(q+pooled))).squeeze(-1)
        return (logits, weights) if return_attention else logits
