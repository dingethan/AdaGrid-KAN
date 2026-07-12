#!/usr/bin/env python3
"""
消融实验：噪声鲁棒性与超参敏感性
====================================
系统化测试 EcoGrow-KAN 在不同噪声水平和超参配置下的表现。
"""

from __future__ import annotations

import argparse
import os
import sys
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kan_layer import AdaptiveKANLayer
from src.grid_scheduler import (
    EcoGrowScheduler,
    PlateauGridExpander,
    count_trainable_parameters,
)


def target_function(x):
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


def make_data(noise_std, seed=42, n_train=160, n_val=160, n_test=400):
    torch.manual_seed(seed)
    x_train = torch.linspace(-1, 1, n_train).unsqueeze(1)
    x_val = torch.linspace(-1, 1, n_val).unsqueeze(1)
    x_test = torch.linspace(-1, 1, n_test).unsqueeze(1)
    y_train_clean = target_function(x_train)
    y_val = target_function(x_val)
    y_test = target_function(x_test)
    y_train = y_train_clean + noise_std * torch.randn_like(y_train_clean)
    return x_train, y_train, x_val, y_val, x_test, y_test


def train_model(model, scheduler_type, x_train, y_train, x_val, y_val,
                device, patience=10, max_grid=64, epochs=1200, lr=0.03,
                trial_epochs=25, min_improvement=0.01, min_efficiency=0.05,
                verbose=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    x_train_d, y_train_d = x_train.to(device), y_train.to(device)
    x_val_d, y_val_d = x_val.to(device), y_val.to(device)

    if scheduler_type == "adagrid":
        scheduler = PlateauGridExpander(model, patience=patience, max_grid=max_grid, verbose=False)
    elif scheduler_type == "ecogrow":
        scheduler = EcoGrowScheduler(
            model, patience=patience, max_grid=max_grid,
            trial_epochs=trial_epochs, min_improvement=min_improvement,
            min_efficiency=min_efficiency, verbose=verbose,
        )
    else:
        scheduler = None

    for epoch in range(epochs):
        model.train()
        pred = model(x_train_d)
        loss = criterion(pred, y_train_d)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_d), y_val_d).item()

        if scheduler is not None and scheduler_type == "adagrid":
            expanded, _ = scheduler.step(loss.item(), epoch=epoch)
            if expanded:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        elif scheduler is not None and scheduler_type == "ecogrow":
            result = scheduler.step(train_loss=loss.item(), val_loss=val_loss, epoch=epoch)
            model = result.model
            if result.optimizer_reset_required:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.eval()
    with torch.no_grad():
        test_pred = model(x_val.to(device))
        test_mse = criterion(test_pred, y_val.to(device)).item()

    return {
        "test_mse": test_mse,
        "n_params": count_trainable_parameters(model),
        "final_grid": model.grid_size,
    }


# ==========================================================================
# 实验 1：噪声鲁棒性
# ==========================================================================

def run_noise_robustness(seeds, device, args):
    """在不同噪声水平下对比各方法。"""
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    all_results = {nl: {"fixed": [], "adagrid": [], "ecogrow": []} for nl in noise_levels}

    for noise_std in noise_levels:
        print(f"\n  噪声水平: {noise_std}")
        for seed in seeds:
            data = make_data(noise_std, seed=seed)

            # Fixed KAN
            model_f = AdaptiveKANLayer(1, 1, grid_size=args.fixed_grid, spline_order=3)
            r_f = train_model(model_f, "fixed", *data, device, epochs=args.epochs, verbose=False)
            all_results[noise_std]["fixed"].append(r_f)

            # AdaGrid
            model_a = AdaptiveKANLayer(1, 1, grid_size=args.initial_grid, spline_order=3)
            r_a = train_model(model_a, "adagrid", *data, device,
                              patience=args.patience, max_grid=args.max_grid, epochs=args.epochs)
            all_results[noise_std]["adagrid"].append(r_a)

            # EcoGrow
            model_e = AdaptiveKANLayer(1, 1, grid_size=args.initial_grid, spline_order=3)
            r_e = train_model(model_e, "ecogrow", *data, device,
                              patience=args.patience, max_grid=args.max_grid, epochs=args.epochs,
                              trial_epochs=args.trial_epochs)
            all_results[noise_std]["ecogrow"].append(r_e)

    return noise_levels, all_results


def plot_noise_robustness(noise_levels, all_results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"fixed": "#4a6cf7", "adagrid": "#ff7a1a", "ecogrow": "#20b38a"}
    labels = {"fixed": "Fixed KAN", "adagrid": "AdaGrid-KAN", "ecogrow": "EcoGrow-KAN"}

    noise_arr = np.array(noise_levels)

    # Test MSE
    ax = axes[0]
    for method in ["fixed", "adagrid", "ecogrow"]:
        means = [np.mean([r["test_mse"] for r in all_results[nl][method]]) for nl in noise_levels]
        stds = [np.std([r["test_mse"] for r in all_results[nl][method]]) for nl in noise_levels]
        ax.errorbar(noise_arr, means, yerr=stds, marker='o', color=colors[method],
                    label=labels[method], capsize=3, linewidth=1.5)
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs Noise Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Params
    ax = axes[1]
    for method in ["fixed", "adagrid", "ecogrow"]:
        means = [np.mean([r["n_params"] for r in all_results[nl][method]]) for nl in noise_levels]
        ax.plot(noise_arr, means, marker='s', color=colors[method],
                label=labels[method], linewidth=1.5)
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Parameters vs Noise Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Final Grid
    ax = axes[2]
    for method in ["adagrid", "ecogrow"]:
        means = [np.mean([r["final_grid"] for r in all_results[nl][method]]) for nl in noise_levels]
        ax.plot(noise_arr, means, marker='^', color=colors[method],
                label=labels[method], linewidth=1.5)
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("Final Grid Size")
    ax.set_title("Final Grid vs Noise Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Noise Robustness Analysis (Multi-Seed)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "noise_robustness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图片已保存至: {path}")


def print_noise_table(noise_levels, all_results):
    print("\n" + "=" * 95)
    print(f"{'Noise':>8} {'Fixed Val MSE':>16} {'AdaGrid Val MSE':>16} {'EcoGrow Val MSE':>16} "
          f"{'EcoGrow Params':>16} {'AdaGrid Params':>16}")
    print("-" * 95)
    for nl in noise_levels:
        for method in ["fixed", "adagrid", "ecogrow"]:
            pass
        f_mse = np.mean([r["test_mse"] for r in all_results[nl]["fixed"]])
        a_mse = np.mean([r["test_mse"] for r in all_results[nl]["adagrid"]])
        e_mse = np.mean([r["test_mse"] for r in all_results[nl]["ecogrow"]])
        e_params = np.mean([r["n_params"] for r in all_results[nl]["ecogrow"]])
        a_params = np.mean([r["n_params"] for r in all_results[nl]["adagrid"]])
        print(f"{nl:>8.2f} {f_mse:>16.6f} {a_mse:>16.6f} {e_mse:>16.6f} "
              f"{e_params:>16.1f} {a_params:>16.1f}")
    print("=" * 95)


# ==========================================================================
# 实验 2：超参敏感性
# ==========================================================================

def run_hyperparam_sensitivity(seeds, device, args):
    """测试 EcoGrow-KAN 对关键超参的敏感性。"""
    configs = {
        "patience": [3, 5, 8, 10, 15, 20],
        "min_improvement": [0.001, 0.005, 0.01, 0.02, 0.05],
        "min_efficiency": [0.01, 0.03, 0.05, 0.1, 0.2],
        "trial_epochs": [10, 15, 20, 25, 30, 40],
    }
    noise_std = 0.15
    all_results = {param: {v: [] for v in vals} for param, vals in configs.items()}

    for param_name, param_vals in configs.items():
        print(f"\n  超参: {param_name}")
        for val in param_vals:
            for seed in seeds:
                data = make_data(noise_std, seed=seed)
                model = AdaptiveKANLayer(1, 1, grid_size=args.initial_grid, spline_order=3)

                kwargs = dict(
                    patience=args.patience, max_grid=args.max_grid,
                    trial_epochs=args.trial_epochs,
                    min_improvement=args.min_improvement,
                    min_efficiency=args.min_efficiency,
                )
                kwargs[param_name] = val

                r = train_model(model, "ecogrow", *data, device, epochs=args.epochs, **kwargs)
                all_results[param_name][val].append(r)

    return configs, all_results


def plot_hyperparam_sensitivity(configs, all_results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metric_colors = {"test_mse": "#e0455b", "n_params": "#4a6cf7", "final_grid": "#20b38a"}

    for idx, (param_name, param_vals) in enumerate(configs.items()):
        ax = axes[idx // 2][idx % 2]

        means_mse = [np.mean([r["test_mse"] for r in all_results[param_name][v]]) for v in param_vals]
        stds_mse = [np.std([r["test_mse"] for r in all_results[param_name][v]]) for v in param_vals]
        means_params = [np.mean([r["n_params"] for r in all_results[param_name][v]]) for v in param_vals]

        ax.errorbar(param_vals, means_mse, yerr=stds_mse, marker='o', color=metric_colors["test_mse"],
                    capsize=3, linewidth=1.5, label="Test MSE")
        ax2 = ax.twinx()
        ax2.plot(param_vals, means_params, marker='s', color=metric_colors["n_params"],
                 linewidth=1.5, linestyle='--', label="Params")
        ax2.set_ylabel("Parameters", color=metric_colors["n_params"])

        ax.set_xlabel(param_name)
        ax.set_ylabel("Test MSE", color=metric_colors["test_mse"])
        ax.set_title(f"Sensitivity: {param_name}")
        ax.tick_params(axis='y', labelcolor=metric_colors["test_mse"])
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

    plt.suptitle("EcoGrow-KAN Hyperparameter Sensitivity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "hyperparam_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图片已保存至: {path}")


def print_hyperparam_table(configs, all_results):
    for param_name, param_vals in configs.items():
        print(f"\n{'='*70}")
        print(f"超参: {param_name}")
        print(f"{'='*70}")
        print(f"{'值':>10} {'Test MSE':>14} {'Std':>10} {'Params':>10} {'Grid':>8}")
        print("-" * 70)
        for v in param_vals:
            mses = [r["test_mse"] for r in all_results[param_name][v]]
            params = [r["n_params"] for r in all_results[param_name][v]]
            grids = [r["final_grid"] for r in all_results[param_name][v]]
            print(f"{v:>10} {np.mean(mses):>14.6f} {np.std(mses):>10.6f} "
                  f"{np.mean(params):>10.1f} {np.mean(grids):>8.1f}")
        print(f"{'='*70}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Noise Robustness & Hyperparameter Sensitivity")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--initial-grid", type=int, default=3)
    parser.add_argument("--max-grid", type=int, default=64)
    parser.add_argument("--fixed-grid", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--trial-epochs", type=int, default=25)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-efficiency", type=float, default=0.05)
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-hyperparam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(os.path.dirname(__file__), "..")
    seeds = args.seeds

    print(f"Device: {device}")
    print(f"Seeds: {seeds}")

    if not args.skip_noise:
        print("\n" + "#" * 60)
        print("# 实验 1：噪声鲁棒性")
        print("#" * 60)
        noise_levels, noise_results = run_noise_robustness(seeds, device, args)
        print_noise_table(noise_levels, noise_results)
        plot_noise_robustness(noise_levels, noise_results, output_dir)

    if not args.skip_hyperparam:
        print("\n" + "#" * 60)
        print("# 实验 2：超参敏感性")
        print("#" * 60)
        configs, hyper_results = run_hyperparam_sensitivity(seeds, device, args)
        print_hyperparam_table(configs, hyper_results)
        plot_hyperparam_sensitivity(configs, hyper_results, output_dir)

    print("\n所有消融实验完成！")


if __name__ == "__main__":
    main()
