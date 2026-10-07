"""Configuration handling.

Every script takes ``--config path/to/config.yaml`` plus optional
``--set section.key=value`` overrides. The resolved configuration is a plain
nested dict; :func:`work_path` builds paths inside ``paths.work_dir`` so that the
directory layout of all pipeline outputs is defined in one place.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _parse_value(raw: str) -> Any:
    """Interpret a command-line override value with YAML semantics."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load the default config, merge a user config on top and apply overrides."""
    with open(DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_value(raw.strip()))
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="YAML configuration file (default: configs/default.yaml)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="Override individual configuration values")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.config, args.overrides)


def work_path(cfg: dict, *parts: str | int) -> Path:
    """Path inside the pipeline working directory (created on demand)."""
    p = Path(cfg["paths"]["work_dir"]).joinpath(*[str(x) for x in parts])
    return p


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config(cfg: dict, out_path: str | Path) -> None:
    """Store the resolved configuration next to the outputs it produced."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(copy.deepcopy(cfg), fh, sort_keys=False)
