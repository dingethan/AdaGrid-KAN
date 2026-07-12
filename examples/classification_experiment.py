#!/usr/bin/env python3
"""
分类任务实验
============
在合成分类数据集上对比 KAN 与 MLP 的分类表现。
KAN 用于特征变换 + 线性分类头。
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kan_layer import AdaptiveKANLayer
from src.grid_scheduler import (
    EcoGrowScheduler,
    PlateauGridExpander,
    count_trainable_parameters,
)


# ==========================================================================
# 模型定义
# ==========================================================================

class KANClassifier(nn.Module):
    """KAN 特征变换 + 线性分类头。"""
    def __init__(self, input_dim, n_classes, grid_size=3, spline_order=3,
                 hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, n_classes)
        self.kan = AdaptiveKANLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        features = self.kan(x)
        return self.classifier(features)

    @property
    def grid_size(self):
        return self.kan.grid_size

    @property
    def n_params(self):
        return count_trainable_parameters(self)


class MLPClassifier(nn.Module):
    """MLP baseline。"""
    def __init__(self, input_dim, n_classes, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================================
# 数据集
# ==========================================================================

def make_classification_datasets(seed=42):
    """生成多个合成分类数据集。"""
    datasets = {}

    # 1. Moon shape
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=seed)
    datasets["moons"] = (X, y)

    # 2. Circles
    X, y = make_circles(n_samples=1000, noise=0.15, factor=0.5, random_state=seed)
    datasets["circles"] = (X, y)

    # 3. Linearly separable (2D)
    X, y = make_classification(
        n_samples=1000, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1, random_state=seed,
    )
    datasets["linear_2d"] = (X, y)

    # 4. Higher dimensional
    X, y = make_classification(
        n_samples=1000, n_features=8, n_redundant=2,
        n_informative=5, n_clusters_per_class=2, random_state=seed,
    )
    datasets["highdim_8d"] = (X, y)

    return datasets


def to_tensors(X_np, y_np, seed=42):
    """转换为 tensor 并归一化到 [-1,1]。"""
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    # 归一化 X 到 [-1, 1]
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_np)
    X_min, X_max = X_norm.min(), X_norm.max()
    X_range = X_max - X_min if X_max - X_min > 1e-8 else 1.0
    X_tensor = torch.tensor((X_norm - X_min) / X_range * 2 - 1, dtype=torch.float32)

    return X_tensor, y


# ==========================================================================
# 训练与评估
# ==========================================================================

def train_classifier(model, train_loader, val_loader, device, args,
                      scheduler=None, scheduler_type=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs, val_losses = [], [], []
    grid_history = []

    for epoch in range(args.epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.shape[0]
        train_accs.append(correct / max(total, 1))

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss_total, val_correct, val_total = 0.0, 0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_total += criterion(logits, yb).item()
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += yb.shape[0]
            val_accs.append(val_correct / max(val_total, 1))
            val_losses.append(val_loss_total / max(val_total, 1))

        # Scheduler
        if scheduler is not None and scheduler_type == "adagrid":
            expanded, _ = scheduler.step(val_losses[-1], epoch=epoch)
            if expanded:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        elif scheduler is not None and scheduler_type == "ecogrow":
            result = scheduler.step(
                train_loss=1.0 - train_accs[-1], val_loss=val_losses[-1], epoch=epoch,
            )
            if result.optimizer_reset_required:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if hasattr(model, 'grid_size'):
            grid_history.append(model.grid_size)
        else:
            grid_history.append(-1)

    return {
        "train_accs": train_accs,
        "val_accs": val_accs,
        "val_losses": val_losses,
        "grid_history": grid_history,
    }


def evaluate_classifier(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.shape[0]
    return correct / max(total, 1)


def run_classification_experiment(ds_name, X, y, args, device):
    """在单个分类数据集上运行四种方法对比。"""
    input_dim = X.shape[1]
    n_classes = len(torch.unique(y))
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.numpy(), y.numpy(), test_size=0.4, random_state=args.seed, stratify=y.numpy(),
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp,
    )

    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_val_t, y_val_t = to_tensors(X_val, y_val)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    batch_size = min(64, len(X_train_t))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=len(X_val_t))
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=len(X_test_t))

    results = {}
    hidden_dim = max(input_dim * 4, 16)

    print(f"\n{'='*60}")
    print(f"分类数据集: {ds_name} (n={len(X)}, dim={input_dim}, classes={n_classes})")
    print(f"{'='*60}")

    # 1. MLP
    print(f"  训练 MLP (hidden={hidden_dim})...")
    mlp = MLPClassifier(input_dim, n_classes, hidden_dim=hidden_dim)
    train_classifier(mlp, train_loader, val_loader, device, args)
    acc = evaluate_classifier(mlp, test_loader, device)
    results["MLP"] = {"test_acc": acc, "n_params": count_trainable_parameters(mlp)}
    print(f"    Test Acc: {acc:.4f}, Params: {results['MLP']['n_params']}")

    # 2. Fixed KAN
    print(f"  训练 Fixed KAN (G={args.fixed_grid})...")
    fixed = KANClassifier(input_dim, n_classes, grid_size=args.fixed_grid, hidden_dim=hidden_dim)
    train_classifier(fixed, train_loader, val_loader, device, args)
    acc = evaluate_classifier(fixed, test_loader, device)
    results["Fixed KAN"] = {"test_acc": acc, "n_params": count_trainable_parameters(fixed),
                            "final_grid": fixed.grid_size}
    print(f"    Test Acc: {acc:.4f}, Params: {results['Fixed KAN']['n_params']}")

    # 3. AdaGrid-KAN
    print(f"  训练 AdaGrid-KAN (G={args.initial_grid}->{args.max_grid})...")
    ada = KANClassifier(input_dim, n_classes, grid_size=args.initial_grid, hidden_dim=hidden_dim)
    ada_sched = PlateauGridExpander(
        ada.kan, patience=args.patience, max_grid=args.max_grid, verbose=False,
    )
    train_classifier(ada, train_loader, val_loader, device, args,
                     scheduler=ada_sched, scheduler_type="adagrid")
    acc = evaluate_classifier(ada, test_loader, device)
    results["AdaGrid-KAN"] = {"test_acc": acc, "n_params": count_trainable_parameters(ada),
                               "final_grid": ada.grid_size}
    print(f"    Test Acc: {acc:.4f}, Params: {results['AdaGrid-KAN']['n_params']}, "
          f"Grid: {ada.grid_size}")

    # 4. EcoGrow-KAN
    print(f"  训练 EcoGrow-KAN (G={args.initial_grid}->{args.max_grid})...")
    eco = KANClassifier(input_dim, n_classes, grid_size=args.initial_grid, hidden_dim=hidden_dim)
    eco_sched = EcoGrowScheduler(
        eco.kan, patience=args.patience, max_grid=args.max_grid,
        trial_epochs=25, min_improvement=0.01, min_efficiency=0.05, verbose=False,
    )
    train_classifier(eco, train_loader, val_loader, device, args,
                     scheduler=eco_sched, scheduler_type="ecogrow")
    acc = evaluate_classifier(eco, test_loader, device)
    results["EcoGrow-KAN"] = {"test_acc": acc, "n_params": count_trainable_parameters(eco),
                                "final_grid": eco.grid_size}
    print(f"    Test Acc: {acc:.4f}, Params: {results['EcoGrow-KAN']['n_params']}, "
          f"Grid: {eco.grid_size}")

    return results


def plot_classification_results(all_results, output_dir):
    """绘制分类实验结果。"""
    method_colors = {
        "MLP": "#888888", "Fixed KAN": "#4a6cf7",
        "AdaGrid-KAN": "#ff7a1a", "EcoGrow-KAN": "#20b38a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart: test accuracy per dataset
    ax = axes[0]
    datasets = list(all_results.keys())
    n_ds = len(datasets)
    n_methods = 4
    method_names = ["MLP", "Fixed KAN", "AdaGrid-KAN", "EcoGrow-KAN"]
    x = np.arange(n_ds)
    width = 0.18

    for i, method in enumerate(method_names):
        accs = [all_results[ds][method]["test_acc"] for ds in datasets]
        ax.bar(x + i * width, accs, width, color=method_colors[method],
               label=method, alpha=0.85)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classification Accuracy Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.05)

    # Params comparison
    ax = axes[1]
    for i, method in enumerate(method_names):
        params = [all_results[ds][method]["n_params"] for ds in datasets]
        ax.bar(x + i * width, params, width, color=method_colors[method],
               label=method, alpha=0.85)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("KAN Classification Experiments", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "classification_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图片已保存至: {path}")


def print_classification_table(all_results):
    print("\n" + "=" * 100)
    header = (f"{'数据集':<18} {'方法':<16} {'Test Acc':>10} {'Params':>10} {'Final Grid':>12}")
    print(header)
    print("-" * 100)
    for ds_name, results in all_results.items():
        for method, r in results.items():
            grid = r.get("final_grid", "-")
            print(f"{ds_name:<18} {method:<16} {r['test_acc']:>10.4f} {r['n_params']:>10} "
                  f"{grid if isinstance(grid, str) else f'{grid}':>12}")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Classification Experiments for KAN")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--initial-grid", type=int, default=3)
    parser.add_argument("--max-grid", type=int, default=64)
    parser.add_argument("--fixed-grid", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(os.path.dirname(__file__), "..")
    print(f"Device: {device}")

    datasets = make_classification_datasets(seed=args.seed)
    all_results = {}

    for ds_name, (X_np, y_np) in datasets.items():
        results = run_classification_experiment(ds_name, X_np, y_np, args, device)
        all_results[ds_name] = results

    print_classification_table(all_results)
    plot_classification_results(all_results, output_dir)
    print("\n分类实验完成！")


if __name__ == "__main__":
    main()
