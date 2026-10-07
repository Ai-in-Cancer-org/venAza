#!/usr/bin/env python
"""Step 6 - concatenate adapted CONCH and UNI embeddings along the feature axis.

CONCH (semantic, vision-language pretraining) and UNI (morphological texture,
self-supervised pretraining) carry complementary information; the combined
representation is used by the foundation-model MIL head.
"""
import argparse

import _common  # noqa: F401
import torch

from src.config import add_config_args, config_from_args
from src.pipeline import features_dir
from src.utils import get_logger

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--cohort", choices=["internal", "external"], default="internal")
    args = ap.parse_args()
    cfg = config_from_args(args)
    a, b = features_dir(cfg, "conch", args.cohort), features_dir(cfg, "uni", args.cohort)
    out = features_dir(cfg, "combined", args.cohort)
    out.mkdir(parents=True, exist_ok=True)
    common = sorted({p.stem for p in a.glob("*.pt")} & {p.stem for p in b.glob("*.pt")})
    dim = None
    for pid in common:
        fa = torch.load(a / f"{pid}.pt", weights_only=False)
        fb = torch.load(b / f"{pid}.pt", weights_only=False)
        if fa["tile_paths"] != fb["tile_paths"]:
            raise ValueError(f"tile order differs between encoders for patient {pid}")
        feats = torch.cat([fa["features"], fb["features"]], dim=1)
        dim = feats.shape[1]
        torch.save({"features": feats, "tile_paths": fa["tile_paths"], "patient_id": pid},
                   out / f"{pid}.pt")
    log.info(f"{len(common)} patients, combined dimension {dim} -> {out}")


if __name__ == "__main__":
    main()
