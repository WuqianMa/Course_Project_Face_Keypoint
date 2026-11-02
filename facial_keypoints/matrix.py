"""Matrix for the facial keypoint study."""

import itertools
import json
import os
import time
import traceback

import yaml

from .paths import PROJECT_ROOT
from .training import train_one_run

_REQUIRED_AXES = ("dataset", "model", "strategy", "seed")


def _make_key(cell):
    return (
        f"{cell['dataset']}__{cell['model']}__"
        f"{cell['strategy']}__seed{int(cell['seed'])}"
    )


def expand_matrix(spec):
    cells = []
    seen = set()

    mat = spec.get("matrix") or {}
    if mat:
        missing = [a for a in _REQUIRED_AXES if a not in mat]
        if missing:
            raise ValueError(
                f"matrix missing required axes: {missing}. "
                f"Need all of {_REQUIRED_AXES}."
            )
        axes = [mat[a] for a in _REQUIRED_AXES]
        for ds, md, st, sd in itertools.product(*axes):
            cell = {"dataset": ds, "model": md, "strategy": st, "seed": sd}
            k = _make_key(cell)
            if k not in seen:
                seen.add(k)
                cells.append(cell)

    explicit = spec.get("cells") or []
    for raw in explicit:
        if isinstance(raw, (list, tuple)):
            if len(raw) != 4:
                raise ValueError(
                    f"explicit cell must be len-4 (dataset, model, "
                    f"strategy, seed); got {raw!r}"
                )
            cell = dict(zip(_REQUIRED_AXES, raw))
        elif isinstance(raw, dict):
            missing = [a for a in _REQUIRED_AXES if a not in raw]
            if missing:
                raise ValueError(f"explicit cell missing axes {missing}: {raw!r}")
            cell = {a: raw[a] for a in _REQUIRED_AXES}
        else:
            raise ValueError(f"unrecognised cell entry: {raw!r}")
        k = _make_key(cell)
        if k not in seen:
            seen.add(k)
            cells.append(cell)

    if not cells:
        raise ValueError("no cells found. Provide `matrix:` and/or `cells:` in YAML.")
    return cells


def _filter_cells(cells, only):
    if not only:
        return cells
    toks = [t.lower() for t in only]
    kept = []
    for c in cells:
        k = _make_key(c).lower()
        if all(t in k for t in toks):
            kept.append(c)
    return kept


def run_matrix(yaml_path, only=None, dry_run=False, overwrite=None):
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    if "out_dir" not in spec:
        raise ValueError(f"{yaml_path}: missing top-level `out_dir:` key")
    out_dir = spec["out_dir"]
    defaults = dict(spec.get("defaults") or {})
    if overwrite is not None:
        defaults["overwrite"] = bool(overwrite)

    cells = expand_matrix(spec)
    cells = _filter_cells(cells, only)

    out_dir_abs = (
        out_dir
        if os.path.isabs(out_dir)
        else os.path.abspath(os.path.join(PROJECT_ROOT, out_dir))
    )
    os.makedirs(out_dir_abs, exist_ok=True)

    print(f"[run_matrix] yaml       = {yaml_path}")
    print(f"[run_matrix] out_dir    = {out_dir_abs}")
    print(f"[run_matrix] n_cells    = {len(cells)}")
    print(f"[run_matrix] defaults   = {defaults}")
    for i, c in enumerate(cells, 1):
        print(f"  [{i:3d}/{len(cells)}] {_make_key(c)}")

    if dry_run:
        print("[run_matrix] --dry-run: not training.")
        return []

    manifest_path = os.path.join(out_dir_abs, "manifest.json")
    manifest = []
    t0_all = time.time()

    for i, cell in enumerate(cells, 1):
        cfg = {**defaults, **cell, "out_dir": out_dir_abs}
        key = _make_key(cell)
        entry = {
            "i": i,
            "n": len(cells),
            "key": key,
            "cfg": cfg,
            "status": "pending",
            "elapsed_sec": None,
            "best_val_mse": None,
            "error": None,
        }
        print(f"\n===== [{i}/{len(cells)}] {key} =====")
        t0 = time.time()
        try:
            summary = train_one_run(cfg)
            entry["status"] = "ok"
            entry["best_val_mse"] = summary.get("best_val_mse")
            entry["result_path"] = summary.get("result_path")
            entry["ckpt_path"] = summary.get("ckpt_path")
            entry["log_path"] = summary.get("log_path")
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"[run_matrix] ERROR on {key}: {entry['error']}")
        entry["elapsed_sec"] = round(time.time() - t0, 2)
        manifest.append(entry)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    total = round(time.time() - t0_all, 2)
    n_ok = sum(1 for e in manifest if e["status"] == "ok")
    n_err = sum(1 for e in manifest if e["status"] == "error")
    print(
        f"\n[run_matrix] done: {n_ok}/{len(cells)} ok, {n_err} error, "
        f"total {total:.1f}s. Manifest at {manifest_path}"
    )
    return manifest
