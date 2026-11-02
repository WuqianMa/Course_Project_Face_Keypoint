"""Training for the facial keypoint study."""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .augmentation import AUG_TELEMETRY, STRATEGY_FNS, _reset_telemetry
from .data import make_loaders
from .losses import masked_mse
from .models import MLP2, MODEL_BUILDERS

EPOCHS = 30

BATCH_SIZE = 128

WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _per_cohort_mse(preds, ys, val_flags):
    pred_np = preds.numpy()
    y_np = ys.numpy()
    n, d = pred_np.shape
    is_asym = val_flags["is_asym"]
    is_hard = val_flags["is_hard"]
    is_frontal = (~is_asym) & (~is_hard)
    is_asym_only = is_asym & (~is_hard)
    out = {}
    for name, mask in [
        ("overall", np.ones(n, dtype=bool)),
        ("frontal", is_frontal),
        ("asym", is_asym_only),
        ("hard", is_hard),
    ]:
        if mask.sum() == 0:
            out[name] = {"n": 0, "mse": float("nan"), "rmse_pix": float("nan")}
        else:
            diff = pred_np[mask] - y_np[mask]
            mse = float((diff**2).mean())
            rmse_px = float(np.sqrt(((diff * 48.0) ** 2).mean()))
            out[name] = {"n": int(mask.sum()), "mse": mse, "rmse_pix": rmse_px}
    return out


def _make_text_logger(log_path):
    def _log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    return _log


def train_one(
    model,
    dl_tr,
    dl_va,
    lr,
    epochs,
    tag,
    val_flags,
    log_fn,
    use_mask=False,
    weight_decay=WEIGHT_DECAY,
):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    plain_loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    logs = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        _reset_telemetry()

        model.train()
        tr_sum, tr_n = 0.0, 0
        for batch in dl_tr:
            if use_mask:
                xb, yb, mb = batch
                mb = mb.to(DEVICE, non_blocking=True)
            else:
                xb, yb = batch
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse(pred, yb, mb) if use_mask else plain_loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_mse = tr_sum / tr_n

        model.eval()
        all_preds, all_ys, all_ms = [], [], []
        with torch.no_grad():
            for batch in dl_va:
                if use_mask:
                    xb, yb, mb = batch
                    all_ms.append(mb.detach().cpu())
                else:
                    xb, yb = batch
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                pred = model(xb)
                all_preds.append(pred.detach().cpu())
                all_ys.append(yb.detach().cpu())
        preds_cat = torch.cat(all_preds, 0)
        ys_cat = torch.cat(all_ys, 0)

        if use_mask:
            ms_cat = torch.cat(all_ms, 0)
            diff = (preds_cat - ys_cat) * ms_cat
            denom = ms_cat.sum().clamp(min=1.0)
            va_mse = float((diff.pow(2)).sum() / denom)
            va_rmse_pix = float(((diff * 48.0).pow(2).sum() / denom).sqrt())
            cohort_mse = None
        else:
            cohort_mse = (
                _per_cohort_mse(preds_cat, ys_cat, val_flags)
                if val_flags is not None
                else None
            )
            if cohort_mse is not None:
                va_mse = cohort_mse["overall"]["mse"]
                va_rmse_pix = cohort_mse["overall"]["rmse_pix"]
            else:
                va_mse = float((preds_cat - ys_cat).pow(2).mean())
                va_rmse_pix = float(((preds_cat - ys_cat) * 48.0).pow(2).mean().sqrt())

        sec = time.time() - t0
        telem_compact = {
            "seen": {k: AUG_TELEMETRY[k] for k in ("frontal", "asym", "hard")},
            "g1_applied": dict(AUG_TELEMETRY["g1_applied"]),
            "g2_applied": dict(AUG_TELEMETRY["g2_applied"]),
        }
        log_entry = {
            "epoch": epoch,
            "train_mse": tr_mse,
            "val_mse": va_mse,
            "val_rmse_pix": va_rmse_pix,
            "sec": sec,
            "cohort_val": cohort_mse,
            "aug_telemetry": telem_compact,
        }
        logs.append(log_entry)

        if cohort_mse is not None:
            log_fn(
                f"[{tag}] epoch {epoch:03d}/{epochs} | "
                f"train_mse {tr_mse:.5f} | val_mse {va_mse:.5f} | "
                f"rmse_pix {va_rmse_pix:.3f} | "
                f"frontal/asym/hard rmse="
                f"{cohort_mse['frontal']['rmse_pix']:.2f}/"
                f"{cohort_mse['asym']['rmse_pix']:.2f}/"
                f"{cohort_mse['hard']['rmse_pix']:.2f} "
                f"(n={cohort_mse['frontal']['n']}/"
                f"{cohort_mse['asym']['n']}/{cohort_mse['hard']['n']}) | "
                f"{sec:.1f}s"
            )
        else:
            log_fn(
                f"[{tag}] epoch {epoch:03d}/{epochs} | "
                f"train_mse {tr_mse:.5f} | val_mse {va_mse:.5f} | "
                f"val_rmse_pix {va_rmse_pix:.3f} | {sec:.1f}s"
            )

        if va_mse < best_val:
            best_val = va_mse
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    return best_val, best_state, logs


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs(out_dir):
    for sub in ("ckpt", "logs", "results"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def train_one_run(cfg):
    """Execute one (dataset x model x strategy x seed) cell."""
    cfg = {**cfg, **cfg.get("hparams", {})}
    ds_name = cfg["dataset"]
    model_name = cfg["model"]
    strategy = cfg["strategy"]
    seed = int(cfg["seed"])
    out_dir = cfg["out_dir"]
    epochs = int(cfg.get("epochs", EPOCHS))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    weight_decay = float(cfg.get("weight_decay", WEIGHT_DECAY))
    hflip_sync = bool(cfg.get("hflip_mask_sync", True))
    num_work = int(cfg.get("num_workers", 0))
    overwrite = bool(cfg.get("overwrite", False))

    assert model_name in MODEL_BUILDERS, f"unknown model: {model_name}"
    if strategy == "none":
        transform_fn = None
    else:
        assert strategy in STRATEGY_FNS, f"unknown strategy: {strategy}"
        transform_fn = STRATEGY_FNS[strategy]

    _ensure_dirs(out_dir)
    key = f"{ds_name}__{model_name}__{strategy}__seed{seed}"
    ckpt_path = os.path.join(out_dir, "ckpt", f"{key}.pt")
    log_path = os.path.join(out_dir, "logs", f"{key}.jsonl")
    result_path = os.path.join(out_dir, "results", f"{key}.json")
    text_log_path = os.path.join(out_dir, "training_log.txt")
    log_fn = _make_text_logger(text_log_path)

    if os.path.exists(result_path) and not overwrite:
        log_fn(f"[skip] {key} (already done). Set overwrite=True to rerun.")
        with open(result_path) as f:
            return json.load(f)

    _seed_all(seed)
    loaders = make_loaders(
        ds_name,
        transform_fn=transform_fn,
        batch_size=batch_size,
        num_workers=num_work,
        hflip_mask_sync=hflip_sync,
    )
    dl_tr, dl_va, n_tr, n_va, tr_cohort_frac, val_flags = loaders
    use_mask = ds_name.startswith("full")

    log_fn(
        f"\n[run] {key} | device={DEVICE} | epochs={epochs} | bs={batch_size} | "
        f"lr={float(cfg.get('lr', MODEL_BUILDERS[model_name][1])):g} | "
        f"use_mask={use_mask} | hflip_mask_sync={hflip_sync}"
    )
    log_fn(
        f"[data] train={n_tr} val={n_va} | cohort frac "
        f"frontal/asym/hard = "
        f"{tr_cohort_frac['frontal']:.3f}/{tr_cohort_frac['asym']:.3f}/{tr_cohort_frac['hard']:.3f}"
    )

    sample_y = next(iter(dl_tr))[1]
    num_outputs = sample_y.shape[1]
    builder, lr = MODEL_BUILDERS[model_name]
    lr = float(cfg.get("lr", lr))
    if model_name == "mlp2" and "model_hidden" in cfg:
        model = MLP2(num_outputs, hidden=int(cfg["model_hidden"]))
    else:
        model = builder(num_outputs)

    t_start = time.time()
    best_val, best_state, logs = train_one(
        model,
        dl_tr,
        dl_va,
        lr=lr,
        epochs=epochs,
        tag=key,
        val_flags=val_flags,
        log_fn=log_fn,
        use_mask=use_mask,
        weight_decay=weight_decay,
    )
    elapsed = time.time() - t_start

    torch.save(best_state, ckpt_path)
    log_fn(f"[ckpt] saved {ckpt_path}")

    with open(log_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    summary = {
        "key": key,
        "dataset": ds_name,
        "model": model_name,
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "hparams": cfg.get("hparams", {}),
        "device": DEVICE,
        "use_mask": use_mask,
        "hflip_mask_sync": hflip_sync,
        "n_train": n_tr,
        "n_val": n_va,
        "tr_cohort_frac": tr_cohort_frac,
        "best_val_mse": best_val,
        "ckpt_path": ckpt_path,
        "log_path": log_path,
        "result_path": result_path,
        "elapsed_sec": elapsed,
    }
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_fn(f"[done] {key} best_val_mse={best_val:.6f}  elapsed={elapsed:.1f}s")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return summary
