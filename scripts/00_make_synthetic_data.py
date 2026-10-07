#!/usr/bin/env python
"""Generate a tiny synthetic dataset to smoke-test the pipeline end to end.

Creates random 'smear-like' ROI images (pink background, purple nuclei-like
blobs) and a matching random clinical table. The data carry no biological
meaning; results on them are meaningless by construction.

    python scripts/00_make_synthetic_data.py --out demo_data --n 30 --n_external 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def roi(rng, w, h, density):
    img = Image.new("RGB", (w, h), tuple(int(v) for v in rng.normal([235, 215, 225], 5)))
    d = ImageDraw.Draw(img)
    for _ in range(int(density * w * h / 2000)):
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(4, 12)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 150, 170))       # erythrocyte-like
    for _ in range(int(density * w * h / 6000)):
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(8, 20)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.normal([90, 50, 140], 15)))
    if rng.random() < 0.5:                                                   # empty area
        d.rectangle([0, 0, w // 3, h // 2], fill=(240, 235, 238))
    return img


def cohort(rng, out: Path, n: int, prefix: str):
    rows = []
    for i in range(n):
        pid = f"{prefix}{i:03d}"
        label = int(rng.random() < 0.75)
        (out / "rois" / pid).mkdir(parents=True, exist_ok=True)
        for k in range(2):
            roi(rng, 358 * 3 + int(rng.integers(0, 40)), 358 * 2 + int(rng.integers(0, 40)),
                1.5 if label else 1.0).save(out / "rois" / pid / f"roi_{k}.png")
        rows.append({
            "patient_id": pid,
            "best_response": rng.choice(["CR", "CRi"]) if label else rng.choice(["PR", "Refractory"]),
            "eln2024_li": rng.choice(["Favorable", "Intermediate", "Adverse"]),
            "swog_cyto": rng.choice(["Favorable", "Intermediate", "Unfavorable", "Unknown"]),
            "age_at_diagnosis": int(rng.integers(40, 90)),
            "bm_blasts": int(rng.integers(20, 95)),
            "sex": rng.choice(["M", "F"]),
            **{m: rng.choice(["Y", "N"], p=[0.2, 0.8]) for m in
               ["NPM1", "IDH1_2", "TP53", "ASXL1", "RAS", "PTPN11", "RUNX1", "FLT3_ITD"]},
        })
    pd.DataFrame(rows).to_csv(out / "clinical.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n_external", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cohort(rng, Path(a.out), a.n, "P")
    cohort(rng, Path(a.out) / "external", a.n_external, "E")
    print(f"synthetic data written to {a.out}")


if __name__ == "__main__":
    main()
