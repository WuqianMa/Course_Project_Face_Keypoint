"""Cli for the facial keypoint study."""

import argparse
from importlib import import_module

import torch

from .augmentation import STRATEGY_FNS
from .data import DATASET_REGISTRY
from .losses import masked_mse, plain_mse
from .matrix import run_matrix
from .models import MODEL_BUILDERS
from .training import EPOCHS, train_one_run
from .tuning import SEARCH_SPACES, random_search


def _run_stage(module, function):
    return getattr(import_module(f"facial_keypoints.{module}"), function)()


def _cli_train(args):
    cfg = {
        "dataset": args.dataset,
        "model": args.model,
        "strategy": args.strategy,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "epochs": args.epochs,
        "overwrite": args.overwrite,
    }
    train_one_run(cfg)


def _cli_matrix(args):
    run_matrix(
        yaml_path=args.yaml_path,
        only=args.only,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def _cli_model_info(_args):
    x = torch.zeros(2, 1, 96, 96)
    for name, (builder, lr) in MODEL_BUILDERS.items():
        if name == "resnet18_pretrained":
            print(
                f"[model_zoo] {name:22s} lr={lr:.0e}  (skipping build; needs internet)"
            )
            continue
        m = builder(8)
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(
            f"[model_zoo] {name:22s} lr={lr:.0e}  "
            f"out={tuple(y.shape)}  params={n_params:,}"
        )


def _cli_loss_test(_args):
    torch.manual_seed(0)
    pred = torch.randn(4, 6)
    y = torch.randn(4, 6)
    m = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss_masked = masked_mse(pred, y, m).item()
    ref = float(((pred - y).pow(2) * m).sum() / m.sum().clamp(min=1.0))
    assert abs(loss_masked - ref) < 1e-7, (loss_masked, ref)
    print(f"[losses] masked_mse OK  loss={loss_masked:.6f}  ref={ref:.6f}")
    lp = plain_mse()(pred, y).item()
    lref = float((pred - y).pow(2).mean())
    assert abs(lp - lref) < 1e-7, (lp, lref)
    print(f"[losses] plain_mse  OK  loss={lp:.6f}  ref={lref:.6f}")


def _cli_eda_dup(_args):
    _run_stage("data_prep.audit", "_main_eda_dup")


def _cli_build_data_full(_args):
    _run_stage("data_prep.full", "_main_rebuild_splits")


def _cli_build_data_large(_args):
    _run_stage("data_prep.large", "_main_build_large_dedup")


def _cli_hp_search(args):
    space = SEARCH_SPACES[args.model]
    random_search(
        model=args.model,
        dataset=args.dataset,
        space=space,
        n_trials=args.n_trials,
        seed=args.seed,
        strategy=args.strategy,
        out_dir=args.out_dir,
        epochs=args.epochs,
    )


def _cli_summary(_args):
    _run_stage("analysis.summary", "_main_build_summary")


def _cli_figures(_args):
    _run_stage("analysis.figures", "_main_build_figures")


def _cli_tables(_args):
    _run_stage("analysis.tables", "_main_build_tables")


def _cli_deep_analysis(_args):
    _run_stage("analysis.deep", "_main_build_deep_analysis")


def _cli_hp_figure(_args):
    _run_stage("analysis.hp", "_main_build_hp_figure")


def _cli_aug_figures(_args):
    _run_stage("analysis.augmentation", "_main_build_aug_figures")


def _cli_detail_figures(_args):
    _run_stage("analysis.detail", "_main_build_detail_figures")


def _cli_arch_figure(_args):
    _run_stage("analysis.architecture", "_main_build_arch_figure")


def _cli_all_analysis(_args):
    """Run every analysis step in sequence (summary → figures → tables → deep → hp/aug/detail/arch)."""
    for module, function in (
        ("analysis.summary", "_main_build_summary"),
        ("analysis.figures", "_main_build_figures"),
        ("analysis.tables", "_main_build_tables"),
        ("analysis.deep", "_main_build_deep_analysis"),
        ("analysis.hp", "_main_build_hp_figure"),
        ("analysis.augmentation", "_main_build_aug_figures"),
        ("analysis.detail", "_main_build_detail_figures"),
        ("analysis.architecture", "_main_build_arch_figure"),
    ):
        print(f"\n[all-analysis] running {function}")
        _run_stage(module, function)


def main():
    ap = argparse.ArgumentParser(
        prog="facial-keypoints",
        description="Facial keypoint ablation study.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- EDA / data preparation ----
    sub.add_parser(
        "eda-dup", help="Cross-split SHA1 duplicate audit (figures + tables)."
    ).set_defaults(func=_cli_eda_dup)
    sub.add_parser(
        "build-data-full", help="Rebuild data_full clean splits (nondedup + dedup)."
    ).set_defaults(func=_cli_build_data_full)
    sub.add_parser(
        "build-data-large-dedup",
        help="Rebuild data_large clean splits with alpha-rule dedup.",
    ).set_defaults(func=_cli_build_data_large)

    # ---- Core training ----
    p_tr = sub.add_parser(
        "train", help="Train a single (dataset, model, strategy, seed) cell."
    )
    p_tr.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p_tr.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    p_tr.add_argument(
        "--strategy", required=True, choices=["none"] + list(STRATEGY_FNS.keys())
    )
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--out_dir", default="./runs")
    p_tr.add_argument("--epochs", type=int, default=EPOCHS)
    p_tr.add_argument("--overwrite", action="store_true")
    p_tr.set_defaults(func=_cli_train)

    p_mx = sub.add_parser("matrix", help="Run a YAML-defined matrix of cells.")
    p_mx.add_argument("yaml_path")
    p_mx.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Keep only cells whose key contains all given tokens.",
    )
    p_mx.add_argument("--dry-run", action="store_true")
    p_mx.add_argument("--overwrite", action="store_true", default=None)
    p_mx.set_defaults(func=_cli_matrix)

    # ---- HP search ----
    p_hp = sub.add_parser(
        "hp-search", help="Random hyperparameter search for one (dataset, model)."
    )
    p_hp.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p_hp.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()))
    p_hp.add_argument("--n-trials", type=int, default=8, dest="n_trials")
    p_hp.add_argument("--seed", type=int, default=0)
    p_hp.add_argument(
        "--strategy", default="none", choices=["none"] + list(STRATEGY_FNS.keys())
    )
    p_hp.add_argument("--out_dir", default="./runs_hpsearch")
    p_hp.add_argument("--epochs", type=int, default=None)
    p_hp.set_defaults(func=_cli_hp_search)

    # ---- Analysis pipeline ----
    sub.add_parser(
        "summary", help="Aggregate runs/ into long/epoch CSVs."
    ).set_defaults(func=_cli_summary)
    sub.add_parser("figures", help="F1-F5 main figures.").set_defaults(
        func=_cli_figures
    )
    sub.add_parser("tables", help="T1-T5 main tables.").set_defaults(func=_cli_tables)
    sub.add_parser("deep-analysis", help="F6-F11 + T6-T12.").set_defaults(
        func=_cli_deep_analysis
    )
    sub.add_parser("hp-figure", help="F20_hp_search.").set_defaults(func=_cli_hp_figure)
    sub.add_parser("aug-figures", help="F12-F16.").set_defaults(func=_cli_aug_figures)
    sub.add_parser("detail-figures", help="F17-F23.").set_defaults(
        func=_cli_detail_figures
    )
    sub.add_parser(
        "arch-figure", help="F_arch 3-model architecture diagram."
    ).set_defaults(func=_cli_arch_figure)
    sub.add_parser(
        "all-analysis", help="Run every analysis step in sequence."
    ).set_defaults(func=_cli_all_analysis)

    # ---- Diagnostics ----
    sub.add_parser(
        "model-info", help="Print parameter counts of all models."
    ).set_defaults(func=_cli_model_info)
    sub.add_parser(
        "loss-test", help="Sanity-check masked_mse + plain_mse on a toy batch."
    ).set_defaults(func=_cli_loss_test)

    args = ap.parse_args()
    args.func(args)
