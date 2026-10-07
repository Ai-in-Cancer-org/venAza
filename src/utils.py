"""Small shared helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def get_logger(name: str = "venAza") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "cuda"):
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_pid(value) -> str:
    """Canonical string form of a patient identifier.

    Removes surrounding whitespace and the trailing ``.0`` that spreadsheet
    software adds to numeric identifiers. Identifiers are otherwise kept as-is,
    so folder names and table entries must agree.
    """
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)


def list_images(directory: str | Path, extensions) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)
