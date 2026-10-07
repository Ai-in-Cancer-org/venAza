"""Attention rollout (Abnar & Zuidema, 2020) for ViT tile encoders.

Each 224x224 tile is represented by 14x14 = 196 patch tokens (16 px) plus a
class token. Attention matrices of all blocks are averaged over heads, mixed
with the identity to account for residual connections, and multiplied through
the network. The class-token row of the result gives each patch token's
relevance for the tile representation.
"""
from __future__ import annotations

import numpy as np
import torch


class AttentionRollout:
    def __init__(self, encoder, head_fusion: str = "mean"):
        self.encoder = encoder
        self.head_fusion = head_fusion
        self._attn, self._hooks = [], []
        for blk in encoder.vit.blocks:
            self._hooks.append(blk.attn.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        # recompute attention probabilities (fused kernels do not expose them)
        x = inputs[0]
        b, n, c = x.shape
        h = module.num_heads
        qkv = module.qkv(x).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        if hasattr(module, "q_norm"):
            q, k = module.q_norm(q), module.k_norm(k)
        attn = ((q * module.scale) @ k.transpose(-2, -1)).softmax(-1)
        self._attn.append(attn.detach().float().cpu())

    def remove(self):
        for hk in self._hooks:
            hk.remove()

    def _fuse(self, a):
        if self.head_fusion == "max":
            return a.max(1).values
        if self.head_fusion == "min":
            return a.min(1).values
        return a.mean(1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """x: [1, 3, H, W] -> relevance map [H/16, W/16] scaled to [0, 1]."""
        self._attn = []
        self.encoder.encode(x)
        n = self._attn[0].shape[-1]
        eye = torch.eye(n).unsqueeze(0)
        result = eye.clone()
        for a in self._attn:
            a = self._fuse(a)
            a = 0.5 * a + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            result = a @ result
        scores = result[0, 0, 1:].numpy()
        g = int(round(np.sqrt(scores.size)))
        scores = scores[: g * g].reshape(g, g)
        rng = scores.max() - scores.min()
        return (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)


def overlay(tile_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.5, cmap="inferno"):
    """Upsample a [g, g] relevance map to the tile size and blend it."""
    import matplotlib
    from PIL import Image
    h, w = tile_rgb.shape[:2]
    up = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    colored = matplotlib.colormaps[cmap](up)[..., :3] * 255
    return (tile_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
