"""Patch extraction with region-adaptive quality control (QC).

For every manually selected region of interest (ROI):

1. The ROI is cropped (never padded) to the nearest integer multiple of the
   patch size; at most ``max_border_crop`` pixels are discarded per dimension.
2. The ROI is partitioned into non-overlapping square patches.
3. Hue, saturation and luminance statistics (mean, standard deviation) are
   computed for the whole ROI. Filtering thresholds are derived from these
   ROI-specific statistics instead of fixed global values, which compensates for
   slide-to-slide differences in staining intensity and specimen age.
4. A pixel counts as *foreground* (nucleated / cellular material) if it is more
   saturated than ``mean_S + sat_k * std_S`` **or** darker than
   ``mean_L + lum_k * std_L``. Patches whose foreground fraction is below
   ``min_foreground`` - i.e. cell-free background or erythrocyte-dominated
   areas - are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class QCParams:
    patch_size: int = 358
    max_border_crop: int = 50
    sat_k: float = 0.5
    lum_k: float = 0.2
    min_foreground: float = 0.16

    @classmethod
    def from_config(cls, cfg: dict) -> "QCParams":
        pe = cfg["patch_extraction"]
        return cls(pe["patch_size"], pe["max_border_crop"], pe["sat_k"],
                   pe["lum_k"], pe["min_foreground"])


def rgb_to_hsl_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised hue, saturation (HSV definition) and luminance for float RGB in [0, 1].

    Luminance is the channel mean (intensity), which is the quantity the
    darkness criterion operates on.
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    delta = maxc - minc

    sat = np.zeros_like(maxc)
    nz = maxc > 0
    sat[nz] = delta[nz] / maxc[nz]

    hue = np.zeros_like(maxc)
    m = delta > 0
    safe = np.where(m, delta, 1.0)
    rc = (g - b) / safe
    gc = 2.0 + (b - r) / safe
    bc = 4.0 + (r - g) / safe
    hue = np.where(m & (maxc == r), rc, hue)
    hue = np.where(m & (maxc == g) & (maxc != r), gc, hue)
    hue = np.where(m & (maxc == b) & (maxc != r) & (maxc != g), bc, hue)
    hue = (hue / 6.0) % 1.0

    lum = img.mean(axis=-1)
    return hue, sat, lum


def roi_statistics(img: np.ndarray) -> dict:
    """ROI-level colour statistics used to derive the adaptive thresholds."""
    hue, sat, lum = rgb_to_hsl_channels(img)
    return {
        "hue_mean": float(hue.mean()), "hue_std": float(hue.std()),
        "sat_mean": float(sat.mean()), "sat_std": float(sat.std()),
        "lum_mean": float(lum.mean()), "lum_std": float(lum.std()),
    }


def adaptive_thresholds(stats: dict, p: QCParams) -> tuple[float, float]:
    sat_thr = stats["sat_mean"] + p.sat_k * stats["sat_std"]
    lum_thr = stats["lum_mean"] + p.lum_k * stats["lum_std"]
    return sat_thr, lum_thr


def foreground_fraction(patch: np.ndarray, sat_thr: float, lum_thr: float) -> float:
    _, sat, lum = rgb_to_hsl_channels(patch)
    fg = (sat > sat_thr) | (lum < lum_thr)
    return float(fg.mean())


def crop_to_grid(img: np.ndarray, patch_size: int, max_border_crop: int) -> np.ndarray:
    h, w = img.shape[:2]
    h_c, w_c = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    if h - h_c > max_border_crop or w - w_c > max_border_crop:
        raise ValueError(
            f"ROI of size {h}x{w} would lose {h - h_c}x{w - w_c} px when cropped to a "
            f"multiple of {patch_size}; limit is {max_border_crop}. Re-export the ROI "
            f"with dimensions close to a multiple of the patch size.")
    if h_c == 0 or w_c == 0:
        raise ValueError(f"ROI {h}x{w} is smaller than one patch ({patch_size}px)")
    return img[:h_c, :w_c]


def extract_roi(roi_path: str | Path, out_dir: str | Path, patient_id: str,
                params: QCParams, rescale_factor: float = 1.0,
                save_rejected: bool = True, save_grid: bool = True,
                patches_root: str | Path | None = None) -> pd.DataFrame:
    """Tile one ROI and apply adaptive QC.

    Patches are written to ``out_dir/<roi>/kept`` and ``out_dir/<roi>/rejected``.
    Returns one row per patch; ``patch_path`` is relative to ``patches_root``.
    """
    roi_path = Path(roi_path)
    roi_name = roi_path.stem
    roi_dir = Path(out_dir) / roi_name
    (roi_dir / "kept").mkdir(parents=True, exist_ok=True)
    if save_rejected:
        (roi_dir / "rejected").mkdir(parents=True, exist_ok=True)
    patches_root = Path(patches_root) if patches_root else Path(out_dir)

    pil = Image.open(roi_path).convert("RGB")
    if rescale_factor != 1.0:
        pil = pil.resize((max(1, round(pil.width * rescale_factor)),
                          max(1, round(pil.height * rescale_factor))), Image.LANCZOS)
    img_u8 = crop_to_grid(np.asarray(pil), params.patch_size, params.max_border_crop)
    img = img_u8.astype(np.float32) / 255.0

    stats = roi_statistics(img)
    sat_thr, lum_thr = adaptive_thresholds(stats, params)

    ps = params.patch_size
    n_rows, n_cols = img.shape[0] // ps, img.shape[1] // ps
    records = []
    for i in range(n_rows):
        for j in range(n_cols):
            y, x = i * ps, j * ps
            patch = img[y:y + ps, x:x + ps]
            frac = foreground_fraction(patch, sat_thr, lum_thr)
            keep = frac >= params.min_foreground
            name = f"patch_{i}_{j}.png"
            sub = "kept" if keep else "rejected"
            path = roi_dir / sub / name
            if keep or save_rejected:
                Image.fromarray(img_u8[y:y + ps, x:x + ps]).save(path)
            records.append({
                "patient_id": patient_id, "roi": roi_name, "patch_name": name,
                "patch_path": str(path.relative_to(patches_root)) if (keep or save_rejected) else "",
                "row": i, "col": j, "x": x, "y": y, "keep": int(keep),
                "foreground_frac": round(frac, 4),
                "sat_thr": round(sat_thr, 4), "lum_thr": round(lum_thr, 4),
                **{k: round(v, 4) for k, v in stats.items()},
            })

    df = pd.DataFrame(records)
    df.to_csv(roi_dir / f"{roi_name}_patches.csv", index=False)
    if save_grid:
        save_qc_grid(img_u8, df, ps, roi_dir / f"{roi_name}_qc_grid.png")
    return df


def save_qc_grid(img_u8: np.ndarray, df: pd.DataFrame, patch_size: int,
                 out_path: Path, max_side: int = 2000) -> None:
    """Overview figure: every patch labelled K (kept, green) or R (rejected, red)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    h, w = img_u8.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    fig, ax = plt.subplots(figsize=(w * scale / 100, h * scale / 100), dpi=100)
    ax.imshow(img_u8)
    for _, r in df.iterrows():
        color = "lime" if r["keep"] else "red"
        ax.add_patch(Rectangle((r["x"], r["y"]), patch_size, patch_size,
                               fill=False, edgecolor=color, linewidth=1.0))
        ax.text(r["x"] + patch_size / 2, r["y"] + patch_size / 2,
                "K" if r["keep"] else "R", color=color, ha="center", va="center",
                fontsize=max(6, int(patch_size * scale / 6)), weight="bold")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path)
    plt.close(fig)
