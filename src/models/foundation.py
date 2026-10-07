"""Pathology foundation models (CONCH ViT-B/16 and UNI ViT-L/16).

* :func:`load_encoder` returns a wrapper with a uniform ``encode`` method
  (one embedding per tile) for the original or DINO-adapted weights.
* :func:`freeze_except` keeps only the final transformer blocks (and the final
  normalisation layer) trainable for self-supervised domain adaptation.

Model weights are gated on the Hugging Face hub; request access to
MahmoodLab/CONCH and MahmoodLab/UNI and log in with ``huggingface-cli login``.

For testing the pipeline without access to the weights, ``name="debug"`` builds
a small randomly initialised timm ViT with the same interface.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from ..data.datasets import IMAGENET_MEAN, IMAGENET_STD


def fm_transform(tile_size: int = 224):
    return T.Compose([T.Lambda(lambda im: im.convert("RGB")), T.Resize(tile_size),
                      T.CenterCrop(tile_size), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class FMEncoder(nn.Module):
    """Uniform wrapper: ``encode(x) -> [B, D]`` and access to the ViT blocks."""

    def __init__(self, kind: str, module: nn.Module):
        super().__init__()
        self.kind = kind
        self.module = module

    @property
    def vit(self) -> nn.Module:
        """The timm VisionTransformer (``.blocks`` lives here)."""
        return self.module.trunk if self.kind == "conch" else self.module

    def forward(self, x):
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "conch":
            out = self.module(x)            # CONCH visual tower: (embedding, tokens)
            return out[0] if isinstance(out, (tuple, list)) else out
        return self.module.forward_features(x)[:, 0]   # CLS token


def _build(name: str, enc_cfg: dict, cache_dir: str | Path | None) -> FMEncoder:
    if name == "conch" and not enc_cfg.get("debug", False):
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained(enc_cfg.get("arch", "conch_ViT-B-16"),
                                                enc_cfg["hf_id"], cache_dir=str(cache_dir))
        return FMEncoder("conch", model.visual)
    import timm
    if enc_cfg.get("debug", False):
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        return FMEncoder("vit", vit)
    vit = timm.create_model(enc_cfg["hf_id"], pretrained=True, init_values=1e-5,
                            dynamic_img_size=True,
                            cache_dir=str(cache_dir) if cache_dir else None)
    return FMEncoder("vit", vit)


def load_encoder(name: str, cfg: dict, adapted_ckpt: str | Path | None = None,
                 device="cpu") -> FMEncoder:
    """Load CONCH / UNI and optionally apply DINO-adapted weights."""
    enc_cfg = cfg["foundation"]["encoders"][name]
    enc = _build(name, enc_cfg, cfg["paths"].get("hf_cache_dir"))
    if adapted_ckpt is not None:
        ckpt = torch.load(adapted_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("encoder_state_dict") or ckpt.get("teacher_encoder") or ckpt
        missing, unexpected = enc.load_state_dict(state, strict=False)
        if unexpected or len(missing) > 0:
            print(f"[load_encoder:{name}] missing={len(missing)} unexpected={len(unexpected)}")
    return enc.to(device).eval()


def freeze_except(enc: FMEncoder, blocks) -> tuple[int, int]:
    """Freeze everything except the given transformer blocks + final norm."""
    for p in enc.parameters():
        p.requires_grad = False
    n_blocks = len(enc.vit.blocks)
    for b in blocks:
        if b >= n_blocks:
            raise ValueError(f"Encoder has {n_blocks} blocks; cannot unfreeze block {b}")
    pat_blocks = re.compile(r"(^|\.)blocks\.(" + "|".join(str(b) for b in blocks) + r")\.")
    for name, p in enc.named_parameters():
        if pat_blocks.search(name):
            p.requires_grad = True
        elif enc.kind == "conch" and ("ln_post" in name or "ln_final" in name):
            p.requires_grad = True
        elif enc.kind == "vit" and re.match(r"^module\.norm\.", name):
            p.requires_grad = True
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    total = sum(p.numel() for p in enc.parameters())
    return trainable, total
