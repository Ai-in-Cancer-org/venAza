#!/usr/bin/env python
"""Step 1 - tile manually selected ROIs and apply region-adaptive QC.

Input : <roi_dir>/<patient_id>/<roi image files>
Output: <work_dir>/patches/...   (or <work_dir>/external/patches for --cohort external)

    python scripts/01_extract_patches.py --config configs/my_run.yaml
    python scripts/01_extract_patches.py --config configs/my_run.yaml --cohort external
"""
import argparse

import _common  # noqa: F401
import pandas as pd
from tqdm import tqdm

from src.config import add_config_args, config_from_args, dump_config
from src.pipeline import patches_dir
from src.preprocessing.patch_extraction import QCParams, extract_roi
from src.utils import get_logger, list_images, normalize_pid

log = get_logger()


def main():
    ap = add_config_args(argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--cohort", choices=["internal", "external"], default="internal")
    ap.add_argument("--input", default=None,
                    help="ROI directory (default: paths.roi_dir or paths.external_roi_dir)")
    ap.add_argument("--resume", action="store_true", help="skip ROIs already processed")
    args = ap.parse_args()
    cfg = config_from_args(args)

    if args.input:
        roi_root = args.input
    elif args.cohort == "internal":
        roi_root = cfg["paths"]["roi_dir"]
    else:
        roi_root = cfg["paths"]["external_roi_dir"]
    out_root = patches_dir(cfg, args.cohort)
    out_root.mkdir(parents=True, exist_ok=True)
    rescale = cfg["external"]["rescale_factor"] if args.cohort == "external" else 1.0
    params = QCParams.from_config(cfg)
    pe = cfg["patch_extraction"]

    from pathlib import Path
    patients = sorted(d for d in Path(roi_root).iterdir() if d.is_dir())
    log.info(f"{len(patients)} patient folders under {roi_root}")
    frames = []
    for pdir in tqdm(patients, desc="patients"):
        pid = normalize_pid(pdir.name)
        rois = list_images(pdir, pe["image_extensions"])
        if not rois:
            log.info(f"  no ROI images for {pid}")
        for roi in rois:
            csv = out_root / pid / roi.stem / f"{roi.stem}_patches.csv"
            if args.resume and csv.exists():
                frames.append(pd.read_csv(csv, dtype={"patient_id": str}))
                continue
            try:
                frames.append(extract_roi(roi, out_root / pid, pid, params, rescale,
                                          pe["save_rejected"], pe["save_grid_figure"],
                                          patches_root=out_root))
            except ValueError as e:
                log.info(f"  [skip] {pid}/{roi.name}: {e}")
    index = pd.concat(frames, ignore_index=True)
    index.to_csv(out_root / "patch_index.csv", index=False)
    summary = index.groupby("patient_id")["keep"].agg(n_patches="size", n_kept="sum")
    summary.to_csv(out_root / "patch_summary.csv")
    dump_config(cfg, out_root / "config_used.yaml")
    log.info(f"Done: {len(index)} patches, {int(index['keep'].sum())} kept, "
             f"{index['patient_id'].nunique()} patients -> {out_root}")


if __name__ == "__main__":
    main()
