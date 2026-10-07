"""Gated-attention multi-instance learning models.

Attention follows Ilse et al. (2018): two fully connected branches with tanh and
sigmoid activations are multiplied element-wise and projected to one scalar
score per tile. Scores are softmax-normalised over *all* tiles of the patient
and the patient representation is the attention-weighted mean of the tile
features, which is classified by a fully connected head (sigmoid output).
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def build_cnn_encoder(name: str = "resnet34", pretrained: bool = True):
    """ImageNet CNN with the classification layer removed."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.get_model(name, weights=weights)
    if hasattr(model, "fc"):
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        feat_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    return model, feat_dim


class ResNetAttentionMIL(nn.Module):
    """End-to-end model: CNN tile encoder -> gated attention -> classifier.

    The encoder, attention module and classifier are trained jointly with a
    patient-level binary cross-entropy loss.
    """

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True,
                 attn_dim: int = 256, dropout: float = 0.25, temperature: float = 1.5,
                 chunk_size: int = 8):
        super().__init__()
        self.encoder, feat_dim = build_cnn_encoder(backbone, pretrained)
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.attn_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_w = nn.Linear(attn_dim, 1)
        hidden = max(128, feat_dim // 2)
        self.classifier = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, bag: torch.Tensor) -> torch.Tensor:
        """Encode a bag in chunks; halves the chunk size on CUDA OOM."""
        feats, i, n = [], 0, bag.shape[0]
        while i < n:
            cur = min(self.chunk_size, n - i)
            while True:
                try:
                    feats.append(self.encoder(bag[i:i + cur]))
                    i += cur
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or cur == 1:
                        raise
                    torch.cuda.empty_cache()
                    gc.collect()
                    cur = max(1, cur // 2)
        return torch.cat(feats)

    def aggregate(self, feats: torch.Tensor):
        scores = self.attn_w(self.attn_V(feats) * self.attn_U(feats)).squeeze(1)
        attn = torch.softmax(scores / self.temperature, dim=0)
        pooled = (feats * attn.unsqueeze(1)).sum(0)
        return self.classifier(pooled).squeeze(-1), attn

    def forward(self, bag: torch.Tensor):
        return self.aggregate(self.encode(bag))

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class FeatureAttentionMIL(nn.Module):
    """Gated-attention MIL head on pre-computed foundation-model embeddings.

    LayerNorm -> linear projection -> gated attention pooling -> dropout ->
    linear classifier. Image only: no clinical input enters this model.
    """

    def __init__(self, feat_dim: int, proj_dim: int = 512, attn_dim: int = 128,
                 dropout: float = 0.6):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(nn.Linear(feat_dim, proj_dim), nn.ReLU(),
                                  nn.Dropout(dropout * 0.75))
        self.attn_V = nn.Sequential(nn.Linear(proj_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(proj_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor):
        h = self.proj(self.norm(feats))
        attn = F.softmax(self.attn_w(self.attn_V(h) * self.attn_U(h)).squeeze(1), dim=0)
        z = (attn.unsqueeze(1) * h).sum(0)
        return self.classifier(self.dropout(z)).squeeze(-1), attn


class FocalLoss(nn.Module):
    """Binary focal loss; ``alpha`` weights the positive class."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            w = w * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        return (w * bce).mean()
