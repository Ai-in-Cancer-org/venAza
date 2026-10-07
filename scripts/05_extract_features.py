#!/usr/bin/env python
"""Step 5 - tile embeddings with the DINO-adapted CONCH / UNI encoders.

Writes ``features/<encoder>/<patient_id>.pt`` = {features [N, D], tile_paths, patient_id}.
Tiles are processed in sorted path order so that CONCH and UNI embeddings of a
patient are row-aligned for concatenation.

    python scripts/05_extract_features.py --config cfg.yaml --encoder conch
    python scripts/05_extract_features.py --config cfg.yaml --encoder uni --cohort external
"""
import argparse

import _common  # noqa: F401
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.foundation import fm_transform, load_encoder
from src.config import add_config_args, config_from_args
from src.pipeline import adapted_ckpt, features_dir, load_patch_index, patches_dir
from src.utils import get_device, get_logger

log = get_logger()


class Tiles(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]))


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--encoder", choices=["conch", "uni"], required=True)
    ap.add_argument("--cohort", choices=["internal", "external"], default="internal")
    ap.add_argument("--checkpoint", default=None,
                    help="adapted weights (default: <work_dir>/dino/<encoder>/adapted_encoder_final.pt)")
    ap.add_argument("--no_adaptation", action="store_true",
                    help="use the original (non-adapted) weights")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    ckpt = None if args.no_adaptation else (args.checkpoint or adapted_ckpt(cfg, args.encoder))
    enc = load_encoder(args.encoder, cfg, ckpt, device)
    tf = fm_transform(cfg["foundation"]["tile_size"])
    out = features_dir(cfg, args.encoder, args.cohort)
    out.mkdir(parents=True, exist_ok=True)
    root = patches_dir(cfg, args.cohort)
    tiles = load_patch_index(cfg, args.cohort)
    for pid, g in tqdm(tiles.groupby("patient_id"), desc=f"{args.encoder} features"):
        dst = out / f"{pid}.pt"
        if dst.exists() and not args.overwrite:
            continue
        rel = sorted(g["patch_path"])
        dl = DataLoader(Tiles([root / p for p in rel], tf),
                        batch_size=cfg["foundation"]["extract_batch_size"], num_workers=4)
        with torch.no_grad():
            feats = torch.cat([enc.encode(x.to(device)).float().cpu() for x in dl])
        torch.save({"features": feats, "tile_paths": rel, "patient_id": pid}, dst)
    log.info(f"features written to {out}")


if __name__ == "__main__":
    main()
