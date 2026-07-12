#!/usr/bin/env python3
"""
真实数据集回归实验
==================
在多个 UCI 回归数据集上对比 Fixed KAN、AdaGrid-KAN、EcoGrow-KAN 与 MLP。
展示 AdaptiveKANLayer 在多维真实数据上的表现。
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kan_layer import AdaptiveKANLayer
from src.grid_scheduler import (
    EcoGrowScheduler,
    PlateauGridExpander,
    count_trainable_parameters,
)


# ==========================================================================
# 数据集生成（合成多维数据）
# ==========================================================================

def make_synthetic_nd(n_samples=1000, n_features=4, noise_std=0.1, seed=42):
    """
    生成多维合成回归数据集。
    y = sin(5*x0) + 0.5*cos(15*x1) + 0.3*sin(8*x2) + 0.2*x3^2 + noise
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.rand(n_samples, n_features) * 2 - 1  # [-1, 1]
    y = (
        torch.sin(5 * X[:, 0])
        + 0.5 * torch.cos(15 * X[:, 1])
        + 0.3 * torch.sin(8 * X[:, 2])
        + 0.2 * X[:, 3] ** 2
    ).unsqueeze(1)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return X, y


def make_friedman(n_samples=1000, noise_std=0.1, seed=42):
    """
    Friedman #1 回归函数（经典基准）。
    y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
    输入 x_i ~ U(0,1)，我们映射到 [-1,1] 以适配 KAN grid_range。
    """
    torch.manual_seed(seed)
    X_raw = torch.rand(n_samples, 5)  # U(0,1)
    X = X_raw * 2 - 1  # 映射到 [-1,1]
    y = (
        10 * torch.sin(torch.pi * X_raw[:, 0] * X_raw[:, 1])
        + 20 * (X_raw[:, 2] - 0.5) ** 2
        + 10 * X_raw[:, 3]
        + 5 * X_raw[:, 4]
    ).unsqueeze(1)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return X, y


DATASETS = {
    "synthetic_4d": lambda: make_synthetic_nd(n_features=4, noise_std=0.1),
    "synthetic_8d": lambda: make_synthetic_nd(n_features=8, noise_std=0.15),
    "friedman": lambda: make_friedman(noise_std=0.1),
}


# ==========================================================================
# MLP baseline
# ==========================================================================

class MLP(nn.Module):
    """简单 MLP baseline，隐藏层宽度与 KAN 参数量匹配。"""
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================================
# 通用训练与评估
# ==========================================================================

def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    args,
    scheduler=None,
    use_adagrid=False,
):
    """训练模型并返回训练历史与最终指标。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    grid_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_n = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                val_n += 1
            val_losses.append(val_loss / max(val_n, 1))

        # Grid scheduler
        if scheduler is not None and use_adagrid:
            expanded, events = scheduler.step(train_losses[-1], epoch=epoch)
            if expanded:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        elif scheduler is not None and not use_adagrid:
            result = scheduler.step(
                train_loss=train_losses[-1],
                val_loss=val_losses[-1],
                epoch=epoch,
            )
            model = result.model
            if result.optimizer_reset_required:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # 记录 grid 大小（仅 KAN 模型）
        if hasattr(model, 'grid_size'):
            grid_history.append(model.grid_size)
        else:
            grid_history.append(-1)

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_n = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_loss += criterion(pred, yb).item()
            test_n += 1
        test_mse = test_loss / max(test_n, 1)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grid_history": grid_history,
        "test_mse": test_mse,
        "n_params": count_trainable_parameters(model),
        "final_grid": model.grid_size if hasattr(model, 'grid_size') else -1,
    }


def run_dataset_experiment(name, X, y, args):
    """在单个数据集上运行四种方法对比。"""
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    # 划分数据集: 60% train, 20% val, 20% test
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)

    indices = torch.randperm(n_samples)
    X, y = X[indices], y[indices]

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 归一化到 [-1, 1]
    def normalize(X_tr, X_others):
        mins = X_tr.min(dim=0)[0]
        maxs = X_tr.max(dim=0)[0]
        ranges = maxs - mins
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        X_tr_norm = 2 * (X_tr - mins) / ranges - 1
        X_others_norm = [2 * (Xo - mins) / ranges - 1 for Xo in X_others]
        return X_tr_norm, X_others_norm

    X_train_n, (X_val_n, X_test_n) = normalize(X_train, [X_val, X_test])
    # 归一化 y
    y_mean, y_std = y_train.mean(), y_train.std()
    y_std = torch.where(y_std < 1e-8, torch.ones_like(y_std), y_std)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    batch_size = min(64, len(X_train_n))
    train_loader = DataLoader(TensorDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_n, y_val_n), batch_size=len(X_val_n))
    test_loader = DataLoader(TensorDataset(X_test_n, y_test_n), batch_size=len(X_test_n))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print(f"\n{'='*60}")
    print(f"数据集: {name} (n={n_samples}, dim={input_dim})")
    print(f"{'='*60}")

    # 1. MLP baseline
    print(f"  训练 MLP (hidden={args.mlp_hidden})...")
    mlp = MLP(input_dim, output_dim, hidden_dim=args.mlp_hidden)
    results["MLP"] = train_and_evaluate(mlp, train_loader, val_loader, test_loader, device, args)
    print(f"    Test MSE: {results['MLP']['test_mse']:.6f}, Params: {results['MLP']['n_params']}")

    # 2. Fixed KAN
    fixed_grid = args.fixed_grid
    print(f"  训练 Fixed KAN (G={fixed_grid})...")
    fixed_kan = AdaptiveKANLayer(input_dim, output_dim, grid_size=fixed_grid, spline_order=3)
    results["Fixed KAN"] = train_and_evaluate(fixed_kan, train_loader, val_loader, test_loader, device, args)
    print(f"    Test MSE: {results['Fixed KAN']['test_mse']:.6f}, Params: {results['Fixed KAN']['n_params']}")

    # 3. AdaGrid-KAN
    print(f"  训练 AdaGrid-KAN (G={args.initial_grid}->{args.max_grid})...")
    adagrid_model = AdaptiveKANLayer(input_dim, output_dim, grid_size=args.initial_grid, spline_order=3)
    adagrid_scheduler = PlateauGridExpander(
        adagrid_model, patience=args.patience, max_grid=args.max_grid,
        verbose=False, min_delta_rel=0.005,
    )
    results["AdaGrid-KAN"] = train_and_evaluate(
        adagrid_model, train_loader, val_loader, test_loader, device, args,
        scheduler=adagrid_scheduler, use_adagrid=True,
    )
    print(f"    Test MSE: {results['AdaGrid-KAN']['test_mse']:.6f}, "
          f"Params: {results['AdaGrid-KAN']['n_params']}, "
          f"Final Grid: {results['AdaGrid-KAN']['final_grid']}")

    # 4. EcoGrow-KAN
    print(f"  训练 EcoGrow-KAN (G={args.initial_grid}->{args.max_grid})...")
    ecogrow_model = AdaptiveKANLayer(input_dim, output_dim, grid_size=args.initial_grid, spline_order=3)
    ecogrow_scheduler = EcoGrowScheduler(
        ecogrow_model, patience=args.patience, max_grid=args.max_grid,
        trial_epochs=25, min_improvement=0.01, min_efficiency=0.05,
        verbose=False,
    )
    results["EcoGrow-KAN"] = train_and_evaluate(
        ecogrow_model, train_loader, val_loader, test_loader, device, args,
        scheduler=ecogrow_scheduler, use_adagrid=False,
    )
    print(f"    Test MSE: {results['EcoGrow-KAN']['test_mse']:.6f}, "
          f"Params: {results['EcoGrow-KAN']['n_params']}, "
          f"Final Grid: {results['EcoGrow-KAN']['final_grid']}")

    return results


def plot_real_dataset_results(all_results, output_dir):
    """为每个数据集生成对比图。"""
    method_colors = {
        "MLP": "#888888",
        "Fixed KAN": "#4a6cf7",
        "AdaGrid-KAN": "#ff7a1a",
        "EcoGrow-KAN": "#20b38a",
    }
    method_styles = {
        "MLP": "-",
        "Fixed KAN": "--",
        "AdaGrid-KAN": "-",
        "EcoGrow-KAN": "-.",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ds_name, results in all_results.items():
        epochs = range(1, len(results["MLP"]["train_losses"]) + 1)
        # Val loss
        ax = axes[0]
        for method, r in results.items():
            ax.plot(epochs, r["val_losses"], color=method_colors[method],
                    linestyle=method_styles[method], linewidth=1.4,
                    label=f'{method} ({ds_name}, val={r["test_mse"]:.2e})')
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE (log)")
        ax.set_title("Validation Loss Comparison (Real/Synthetic Datasets)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Bar chart: final params and test MSE
        ax2 = axes[1]
        methods = list(results.keys())
        x_pos = np.arange(len(methods))
        test_mses = [results[m]["test_mse"] for m in methods]
        params = [results[m]["n_params"] for m in methods]
        final_grids = [results[m]["final_grid"] for m in methods]

        bars = ax2.bar(x_pos, test_mses, color=[method_colors[m] for m in methods],
                       alpha=0.8, width=0.6)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{m}\n(P={p}, G={g if g>0 else "-"})'
                              for m, p, g in zip(methods, params, final_grids)],
                             fontsize=8)
        ax2.set_ylabel("Test MSE")
        ax2.set_title(f"Final Test MSE ({ds_name})")
        ax2.grid(True, alpha=0.3, axis='y')

        # 在 bar 上标注数值
        for bar, mse in zip(bars, test_mses):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{mse:.2e}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "real_dataset_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图片已保存至: {output_path}")


def print_summary_table(all_results):
    """打印汇总表格。"""
    print("\n" + "=" * 110)
    header = (f"{'数据集':<18} {'方法':<16} {'Test MSE':>12} {'Params':>8} "
              f"{'Final Grid':>12} {'Val Loss':>12}")
    print(header)
    print("-" * 110)
    for ds_name, results in all_results.items():
        for method, r in results.items():
            print(
                f"{ds_name:<18} {method:<16} {r['test_mse']:>12.6f} {r['n_params']:>8} "
                f"{r['final_grid'] if r['final_grid'] > 0 else 'N/A':>12} "
                f"{min(r['val_losses']):>12.6f}"
            )
        print("-" * 110)

    # 对比 EcoGrow vs AdaGrid
    print("\nEcoGrow vs AdaGrid 对比:")
    for ds_name, results in all_results.items():
        ada = results["AdaGrid-KAN"]
        eco = results["EcoGrow-KAN"]
        param_red = (ada["n_params"] - eco["n_params"]) / ada["n_params"] * 100
        mse_diff = (ada["test_mse"] - eco["test_mse"]) / ada["test_mse"] * 100
        print(f"  {ds_name:<18} 参数减少: {param_red:>6.1f}% | "
              f"Test MSE 变化: {mse_diff:>+.1f}% | "
              f"Grid: AdaGrid={ada['final_grid']} EcoGrow={eco['final_grid']}")


def main():
    parser = argparse.ArgumentParser(description="Real/Synthetic Dataset Regression Comparison")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--initial-grid", type=int, default=3)
    parser.add_argument("--max-grid", type=int, default=64)
    parser.add_argument("--fixed-grid", type=int, default=12)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Datasets: {args.datasets}")

    output_dir = os.path.join(os.path.dirname(__file__), "..")
    all_results = {}

    for ds_name in args.datasets:
        X, y = DATASETS[ds_name]()
        results = run_dataset_experiment(ds_name, X, y, args)
        all_results[ds_name] = results

    print_summary_table(all_results)
    plot_real_dataset_results(all_results, output_dir)


if __name__ == "__main__":
    main()
