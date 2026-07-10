#!/usr/bin/env python3
"""
EcoGrow-KAN 三方对比实验
========================
比较 Fixed KAN、AdaGrid-KAN、EcoGrow-KAN 在含噪声训练集上的表现。

目标：展示 EcoGrow 可拒绝收益低的扩容，用更少 Grid/参数获得相近或更好的验证效果。
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid_scheduler import (
    EcoGrowScheduler,
    ExtendGridOnPlateau,
    count_trainable_parameters,
)
from src.kan_layer import DynamicKANLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Fixed KAN, AdaGrid-KAN, and EcoGrow-KAN."
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--trial-epochs", type=int, default=30)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-efficiency", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42,
                        help="Single-run random seed (used when --seeds is not set)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for averaged report, e.g. --seeds 42 43 44 45 46")
    parser.add_argument("--plot-seed", type=int, default=None,
                        help="Which seed to use for saving figures (default: first seed)")
    parser.add_argument("--fixed-grid", type=int, default=12)
    parser.add_argument("--initial-grid", type=int, default=3)
    parser.add_argument("--max-grid", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=160)
    parser.add_argument("--val-size", type=int, default=160)
    parser.add_argument("--test-size", type=int, default=400)
    return parser.parse_args()


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


def make_datasets(
    noise_std: float,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    x_train = torch.linspace(-1, 1, train_size).unsqueeze(1)
    x_val = torch.linspace(-1, 1, val_size).unsqueeze(1)
    x_test = torch.linspace(-1, 1, test_size).unsqueeze(1)

    y_train_clean = target_function(x_train)
    y_val = target_function(x_val)
    y_test = target_function(x_test)

    if noise_std > 0:
        y_train = y_train_clean + noise_std * torch.randn_like(y_train_clean)
    else:
        y_train = y_train_clean

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def rebuild_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device))
        return nn.functional.mse_loss(pred, y.to(device)).item()


def train_fixed_kan(
    data: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = DynamicKANLayer(
        in_dim=1, out_dim=1, grid_size=args.fixed_grid,
        spline_order=3, grid_range=(-1.0, 1.0),
    ).to(device)
    optimizer = rebuild_optimizer(model, args.lr, args.weight_decay)

    train_loss_hist, val_loss_hist, grid_hist = [], [], []

    for epoch in range(args.epochs):
        model.train()
        pred = model(data["x_train"].to(device))
        loss = nn.functional.mse_loss(pred, data["y_train"].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        train_loss_hist.append(loss.item())
        val_loss_hist.append(evaluate_mse(model, data["x_val"], data["y_val"], device))
        grid_hist.append(model.grid_size)

    return {
        "name": "Fixed KAN",
        "model": model,
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "grid_hist": grid_hist,
        "events": [],
        "accepted": 0,
        "rejected": 0,
    }


def train_adagrid(
    data: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = DynamicKANLayer(
        in_dim=1, out_dim=1, grid_size=args.initial_grid,
        spline_order=3, grid_range=(-1.0, 1.0),
    ).to(device)
    scheduler = ExtendGridOnPlateau(
        model, patience=args.patience, max_grid=args.max_grid,
        verbose=False, min_delta_rel=0.005,
    )
    optimizer = rebuild_optimizer(model, args.lr, args.weight_decay)

    train_loss_hist, val_loss_hist, grid_hist = [], [], []
    events = []

    for epoch in range(args.epochs):
        model.train()
        pred = model(data["x_train"].to(device))
        loss = nn.functional.mse_loss(pred, data["y_train"].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        train_loss_hist.append(loss.item())
        val_loss_hist.append(evaluate_mse(model, data["x_val"], data["y_val"], device))
        grid_hist.append(model.grid_size)

        refined, refine_events = scheduler.step(loss.item(), epoch=epoch)
        if refined:
            events.extend(refine_events)
            optimizer = rebuild_optimizer(model, args.lr, args.weight_decay)

    return {
        "name": "AdaGrid-KAN",
        "model": model,
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "grid_hist": grid_hist,
        "events": events,
        "accepted": len(events),
        "rejected": 0,
    }


def train_ecogrow(
    data: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)
    model = DynamicKANLayer(
        in_dim=1, out_dim=1, grid_size=args.initial_grid,
        spline_order=3, grid_range=(-1.0, 1.0),
    ).to(device)
    scheduler = EcoGrowScheduler(
        model,
        patience=args.patience,
        max_grid=args.max_grid,
        trial_epochs=args.trial_epochs,
        min_improvement=args.min_improvement,
        min_efficiency=args.min_efficiency,
        verbose=verbose,
    )
    optimizer = rebuild_optimizer(model, args.lr, args.weight_decay)

    train_loss_hist, val_loss_hist, grid_hist = [], [], []

    for epoch in range(args.epochs):
        model.train()
        pred = model(data["x_train"].to(device))
        train_loss = nn.functional.mse_loss(pred, data["y_train"].to(device))
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        val_loss = evaluate_mse(model, data["x_val"], data["y_val"], device)
        train_loss_hist.append(train_loss.item())
        val_loss_hist.append(val_loss)
        grid_hist.append(model.grid_size)

        result = scheduler.step(
            train_loss=train_loss.item(),
            val_loss=val_loss,
            epoch=epoch,
        )
        model = result.model

        if result.optimizer_reset_required:
            optimizer = rebuild_optimizer(model, args.lr, args.weight_decay)

    return {
        "name": "EcoGrow-KAN",
        "model": model,
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "grid_hist": grid_hist,
        "events": scheduler.events,
        "accepted": scheduler.num_accepted,
        "rejected": scheduler.num_rejected,
        "blocked": scheduler.num_blocked,
    }


def summarize_result(result: dict, data: dict[str, torch.Tensor], device: torch.device) -> dict:
    model = result["model"]
    return {
        "name": result["name"],
        "final_grid": model.grid_size,
        "train_mse": evaluate_mse(model, data["x_train"], data["y_train"], device),
        "val_mse": evaluate_mse(model, data["x_val"], data["y_val"], device),
        "test_mse": evaluate_mse(model, data["x_test"], data["y_test"], device),
        "params": count_trainable_parameters(model),
        "accepted": result["accepted"],
        "rejected": result.get("rejected", 0),
        "blocked": result.get("blocked", 0),
    }


def print_results_table(summaries: list[dict], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print("\n" + "=" * 100)
    header = (
        f"{'Method':<16} {'Grid':>6} {'Params':>8} {'Train MSE':>12} "
        f"{'Val MSE':>12} {'Test MSE':>12} {'Accept':>7} {'Reject':>7} {'Block':>6}"
    )
    print(header)
    print("-" * 100)
    for s in summaries:
        print(
            f"{s['name']:<16} {s['final_grid']:>6} {s['params']:>8} "
            f"{s['train_mse']:>12.6f} {s['val_mse']:>12.6f} {s['test_mse']:>12.6f} "
            f"{s['accepted']:>7} {s['rejected']:>7} {s.get('blocked', 0):>6}"
        )
    print("=" * 100)


def print_parameter_comparison(summaries: list[dict]) -> None:
    by_name = {s["name"]: s for s in summaries}
    if "EcoGrow-KAN" not in by_name or "AdaGrid-KAN" not in by_name:
        return
    eco, ada = by_name["EcoGrow-KAN"], by_name["AdaGrid-KAN"]
    param_reduction = (ada["params"] - eco["params"]) / ada["params"] * 100
    val_improvement = (ada["val_mse"] - eco["val_mse"]) / ada["val_mse"] * 100
    print(
        f"\n★ EcoGrow vs AdaGrid (trainable parameters): "
        f"{eco['params']} vs {ada['params']} "
        f"(EcoGrow 少 {param_reduction:.1f}%)"
    )
    print(
        f"★ EcoGrow vs AdaGrid (validation MSE): "
        f"{eco['val_mse']:.6f} vs {ada['val_mse']:.6f} "
        f"(EcoGrow 低 {val_improvement:.1f}%)"
    )
    print(
        f"★ Final Grid: EcoGrow G={eco['final_grid']} vs AdaGrid G={ada['final_grid']}"
    )


def print_ecogrow_events(ecogrow: dict) -> None:
    print("\nEcoGrow 扩容决策:")
    for event in ecogrow["events"]:
        if event["decision"] == "blocked":
            print(
                f"  epoch {event['epoch']}: "
                f"G {event['old_grid']}→{event['new_grid']} → blocked (仅记录一次)"
            )
        else:
            print(
                f"  epoch {event['epoch']}: "
                f"G {event['old_grid']}→{event['new_grid']} → {event['decision']} "
                f"(improve={event['relative_improvement']*100:.2f}%, "
                f"eff={event['efficiency_score']:.3f})"
            )


def run_experiment(
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    verbose: bool,
) -> tuple[list[dict], list[dict], dict[str, torch.Tensor]]:
    data = make_datasets(
        args.noise_std, args.train_size, args.val_size, args.test_size, seed,
    )
    if verbose:
        print("  训练 Fixed KAN / AdaGrid-KAN / EcoGrow-KAN ...")

    fixed = train_fixed_kan(data, device, args, seed)
    adagrid = train_adagrid(data, device, args, seed)
    ecogrow = train_ecogrow(data, device, args, seed, verbose=verbose)
    all_results = [fixed, adagrid, ecogrow]
    summaries = [summarize_result(r, data, device) for r in all_results]
    for s in summaries:
        if s["name"] == "EcoGrow-KAN":
            s["blocked"] = ecogrow.get("blocked", 0)
        else:
            s["blocked"] = 0
    return all_results, summaries, data


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    plot_seed = args.plot_seed if args.plot_seed is not None else seeds[0]
    multi_seed = len(seeds) > 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Epochs={args.epochs}, noise_std={args.noise_std}, "
        f"trial_epochs={args.trial_epochs}, seeds={seeds}"
    )

    output_dir = os.path.join(os.path.dirname(__file__), "..")
    all_summaries_runs: list[list[dict]] = []
    plot_bundle = None

    for i, seed in enumerate(seeds):
        if multi_seed:
            print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        else:
            print("\n" + "=" * 60)
            print("训练 Fixed KAN / AdaGrid-KAN / EcoGrow-KAN ...")
            print("=" * 60)

        verbose = (seed == plot_seed) if multi_seed else True
        all_results, summaries, data = run_experiment(args, device, seed, verbose)
        all_summaries_runs.append(summaries)

        if seed == plot_seed:
            plot_bundle = (all_results, summaries, data)

    if multi_seed:
        print_multi_seed_aggregate(all_summaries_runs, seeds)
    else:
        print_results_table(all_summaries_runs[0], title="单次实验结果")

    if plot_bundle:
        all_results, summaries, data = plot_bundle
        if multi_seed:
            print_results_table(
                summaries,
                title=f"绘图种子 seed={plot_seed} 的详细结果",
            )
        print_parameter_comparison(summaries)
        print_ecogrow_events(all_results[2])
        plot_results(all_results, summaries, data, device, args, output_dir)


def print_multi_seed_aggregate(all_runs: list[list[dict]], seeds: list[int]) -> None:
    methods = all_runs[0][0]["name"], all_runs[0][1]["name"], all_runs[0][2]["name"]
    method_names = [methods[0], methods[1], methods[2]]

    print("\n" + "=" * 100)
    print(f"多种子汇总 (seeds={seeds}, n={len(seeds)})")
    print("=" * 100)
    header = (
        f"{'Method':<16} {'Grid':>14} {'Params':>14} {'Val MSE':>18} {'Test MSE':>18}"
    )
    print(header)
    print("-" * 100)

    for method_idx, name in enumerate(method_names):
        grids = [run[method_idx]["final_grid"] for run in all_runs]
        params = [run[method_idx]["params"] for run in all_runs]
        val_mses = [run[method_idx]["val_mse"] for run in all_runs]
        test_mses = [run[method_idx]["test_mse"] for run in all_runs]

        def fmt_mean_std(values: list[float], sci: bool = False) -> str:
            mean = np.mean(values)
            std = np.std(values)
            if sci:
                return f"{mean:.2e}±{std:.2e}"
            return f"{mean:.1f}±{std:.1f}"

        print(
            f"{name:<16} {fmt_mean_std(grids):>14} {fmt_mean_std(params):>14} "
            f"{fmt_mean_std(val_mses, sci=True):>18} {fmt_mean_std(test_mses, sci=True):>18}"
        )

    eco_runs = [run[2] for run in all_runs]
    ada_runs = [run[1] for run in all_runs]
    param_red = [
        (a["params"] - e["params"]) / a["params"] * 100
        for e, a in zip(eco_runs, ada_runs)
    ]
    val_red = [
        (a["val_mse"] - e["val_mse"]) / a["val_mse"] * 100
        for e, a in zip(eco_runs, ada_runs)
    ]
    print("-" * 100)
    print(
        f"EcoGrow vs AdaGrid — 参数减少: "
        f"{np.mean(param_red):.1f}% ± {np.std(param_red):.1f}%"
    )
    print(
        f"EcoGrow vs AdaGrid — 验证误差降低: "
        f"{np.mean(val_red):.1f}% ± {np.std(val_red):.1f}%"
    )
    print("=" * 100)

    print("\n逐种子明细 (Grid / Params / Val MSE):")
    for seed, run in zip(seeds, all_runs):
        eco, ada = run[2], run[1]
        print(
            f"  seed={seed}: EcoGrow G={eco['final_grid']} params={eco['params']} "
            f"val={eco['val_mse']:.6f} | AdaGrid G={ada['final_grid']} "
            f"params={ada['params']} val={ada['val_mse']:.6f}"
        )


def plot_results(
    results: list[dict],
    summaries: list[dict],
    data: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    epochs = np.arange(1, args.epochs + 1)
    x_test_np = data["x_test"].numpy().flatten()
    y_test_np = data["y_test"].numpy().flatten()
    x_train_np = data["x_train"].numpy().flatten()
    y_train_np = data["y_train"].numpy().flatten()

    colors = {
        "Fixed KAN": "#4a6cf7",
        "AdaGrid-KAN": "#ff7a1a",
        "EcoGrow-KAN": "#20b38a",
    }

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

    # ---- Figure 1: fitting + val loss ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(x_test_np, y_test_np, "k-", linewidth=2.2, alpha=0.55, label="Ground Truth")
    ax.scatter(
        x_train_np, y_train_np, s=12, alpha=0.35, color="#888888",
        label="Noisy Train", zorder=1,
    )
    for result, summary in zip(results, summaries):
        model = result["model"]
        model.eval()
        with torch.no_grad():
            y_pred = model(data["x_test"].to(device)).cpu().numpy().flatten()
        ax.plot(
            x_test_np, y_pred, color=colors[result["name"]], linewidth=1.6,
            label=f'{result["name"]} (G={summary["final_grid"]}, '
                  f'val={summary["val_mse"]:.2e})',
        )
    ax.set_title("Fitting on Test Set")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for result, summary in zip(results, summaries):
        ax.plot(
            epochs, result["val_loss_hist"],
            color=colors[result["name"]], linewidth=1.4,
            label=f'{result["name"]} (final={summary["val_mse"]:.2e})',
        )
    if all(v > 0 for r in results for v in r["val_loss_hist"]):
        ax.set_yscale("log")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "ecogrow_comparison.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存至: {path1}")

    # ---- Figure 2: grid history ----
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(
        epochs, results[1]["grid_hist"], color=colors["AdaGrid-KAN"],
        linewidth=1.6, label="AdaGrid Grid",
    )
    labels_added: set[str] = set()
    ax.plot(
        epochs, results[2]["grid_hist"], color=colors["EcoGrow-KAN"],
        linewidth=1.6, label="EcoGrow Grid",
    )

    for event in results[2]["events"]:
        decision = event["decision"]
        if decision == "accepted":
            marker, color, y_grid = "^", "#20b38a", event["new_grid"]
        elif decision == "rejected":
            marker, color, y_grid = "v", "#e0455b", event["old_grid"]
        elif decision == "blocked":
            marker, color, y_grid = "s", "#888888", event["old_grid"]
        else:
            continue
        label_key = f"EcoGrow {decision}"
        ax.scatter(
            event["epoch"] + 1, y_grid,
            marker=marker, s=80, color=color, zorder=5,
            label=label_key if label_key not in labels_added else "",
        )
        labels_added.add(label_key)

    ax.set_title("Grid Size over Training (EcoGrow decisions marked)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grid G")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "ecogrow_training_history.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练历史图已保存至: {path2}")


if __name__ == "__main__":
    main()
