#!/usr/bin/env python3
"""
AdaGrid-KAN A/B 对比实验（均匀 Grid 自动生长 vs 传统固定 Grid）
================================================================
与 visualization/index.html 一致：
  橙: AdaGrid — G=3 起步，plateau 均匀翻倍 → G=50
  蓝: 传统 KAN — 训练前设定固定均匀 G（3 / 30 / 50），全程不变
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kan_layer import DynamicKANLayer
from src.grid_scheduler import ExtendGridOnPlateau

# ==========================================================================
# 全局配置（对齐浏览器 demo）
# ==========================================================================
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 1200
INITIAL_GRID = 3
MAX_GRID = 50
PATIENCE = 8
LR = 0.03
MIN_DELTA_REL = 0.005
MIN_DELTA_ABS = 1e-8
TRAD_FIXED_GRIDS = [3, 30, 50]

print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}, AdaGrid: G={INITIAL_GRID}→{MAX_GRID}, Patience: {PATIENCE}")


def target_function(x):
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


x_train = torch.linspace(-1, 1, 200).unsqueeze(1)
y_train = target_function(x_train)
x_test = torch.linspace(-1, 1, 500).unsqueeze(1)
y_test = target_function(x_test)
x_train_d = x_train.to(DEVICE)
y_train_d = y_train.to(DEVICE)
criterion = nn.MSELoss()


def rebuild_optimizer(model, lr=LR):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


def train_adagrid(seed=42):
    """AdaGrid：均匀 Grid 自动翻倍。"""
    torch.manual_seed(seed)
    model = DynamicKANLayer(
        in_dim=1, out_dim=1, grid_size=INITIAL_GRID,
        spline_order=3, grid_range=(-1., 1.)
    ).to(DEVICE)
    scheduler = ExtendGridOnPlateau(
        model, patience=PATIENCE, max_grid=MAX_GRID,
        verbose=False, min_delta=MIN_DELTA_ABS, min_delta_rel=MIN_DELTA_REL,
    )
    optimizer = rebuild_optimizer(model)

    loss_history, grid_history, refine_events = [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        pred = model(x_train_d)
        loss = criterion(pred, y_train_d)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        grid_history.append(model.grid_size)

        refined, events = scheduler.step(loss_val, epoch=epoch)
        if refined:
            refine_events.extend(events)
            optimizer = rebuild_optimizer(model)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(DEVICE)).cpu()
    final_mse = np.mean((y_pred.numpy().flatten() - y_test.numpy().flatten()) ** 2)
    return model, loss_history, grid_history, refine_events, y_pred, final_mse


def train_traditional_fixed(seed, fixed_grid):
    """传统 KAN：固定均匀 Grid，训练中不变。"""
    torch.manual_seed(seed)
    model = DynamicKANLayer(
        in_dim=1, out_dim=1, grid_size=fixed_grid,
        spline_order=3, grid_range=(-1., 1.)
    ).to(DEVICE)
    optimizer = rebuild_optimizer(model)

    loss_history, grid_history = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        pred = model(x_train_d)
        loss = criterion(pred, y_train_d)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        loss_history.append(loss.item())
        grid_history.append(model.grid_size)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(DEVICE)).cpu()
    final_mse = np.mean((y_pred.numpy().flatten() - y_test.numpy().flatten()) ** 2)
    return model, loss_history, grid_history, y_pred, final_mse


# ==========================================================================
# 运行
# ==========================================================================
print("\n" + "=" * 60)
print("训练 AdaGrid-KAN（均匀自动翻倍 G=3→50）...")
print("=" * 60)
ada_model, ada_loss, ada_grid_hist, ada_events, ada_pred, ada_mse = train_adagrid(42)
print(f"  完成: MSE={ada_mse:.6f}, Grid={ada_model.grid_size}, 翻倍 {len(ada_events)} 次")
for ep, og, ng in ada_events:
    print(f"    Epoch {ep}: G {og} → {ng}")

trad_results = {}
for g in TRAD_FIXED_GRIDS:
    print("\n" + "=" * 60)
    print(f"训练传统 KAN（固定 G={g}）...")
    print("=" * 60)
    m, loss, gh, pred, mse = train_traditional_fixed(42, g)
    trad_results[g] = dict(model=m, loss=loss, pred=pred, mse=mse)
    print(f"  完成: MSE={mse:.6f}, Grid={m.grid_size}")

# ==========================================================================
# 可视化
# ==========================================================================
x_test_np = x_test.numpy().flatten()
y_test_np = y_test.numpy().flatten()
ada_pred_np = ada_pred.numpy().flatten()
epochs = np.arange(1, NUM_EPOCHS + 1)

trad_colors = {3: '#4a6cf7', 30: '#6b8cff', 50: '#9aa3c4'}
trad_styles = {3: '-', 30: '--', 50: '-.'}

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'font.family': 'DejaVu Sans',
})
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ---- Loss 对比 ----
ax = axes[0, 0]
ax.plot(epochs, ada_loss, color='#ff7a1a', linewidth=1.6, alpha=0.9,
        label=f'AdaGrid auto (MSE={ada_mse:.2e})')
for g in TRAD_FIXED_GRIDS:
    r = trad_results[g]
    ax.plot(epochs, r['loss'], color=trad_colors[g], linestyle=trad_styles[g],
            linewidth=1.4, alpha=0.85, label=f'Trad G={g}, MSE={r["mse"]:.2e}')
for ep, _, _ in ada_events:
    ax.axvline(x=ep, color='#ff7a1a', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_yscale('log')
ax.set_title('Loss: AdaGrid Auto-Grow vs Traditional Fixed Grid')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE (log)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Grid 大小随 epoch ----
ax = axes[0, 1]
ax.plot(epochs, ada_grid_hist, color='#ff7a1a', linewidth=1.8, label='AdaGrid')
for g in TRAD_FIXED_GRIDS:
    ax.axhline(y=g, color=trad_colors[g], linestyle=trad_styles[g],
               linewidth=1.2, alpha=0.8, label=f'Trad G={g}')
ax.set_title('Grid Size over Training')
ax.set_xlabel('Epoch')
ax.set_ylabel('Grid G')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- AdaGrid 拟合 ----
ax = axes[1, 0]
ax.plot(x_test_np, y_test_np, 'k-', linewidth=2.2, alpha=0.55, label='Ground Truth')
ax.plot(x_test_np, ada_pred_np, color='#ff7a1a', linewidth=1.8,
        label=f'AdaGrid G={ada_model.grid_size}')
for x in ada_model.grid.cpu().numpy():
    ax.axvline(x=x, color='#ff7a1a', alpha=0.25, linewidth=0.8, linestyle='--')
ax.set_title(f'AdaGrid-KAN Fit (MSE={ada_mse:.2e}, Grid={ada_model.grid_size})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- 传统 G=3 vs G=50 拟合 ----
ax = axes[1, 1]
ax.plot(x_test_np, y_test_np, 'k-', linewidth=2.2, alpha=0.55, label='Ground Truth')
for g in [3, 50]:
    r = trad_results[g]
    pred_np = r['pred'].numpy().flatten()
    ax.plot(x_test_np, pred_np, color=trad_colors[g], linestyle=trad_styles[g],
        label=f'Trad G={g}, MSE={r["mse"]:.2e}')
ax.set_title('Traditional KAN: Fixed G=3 vs Fixed G=50')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(
    'AdaGrid-KAN: Uniform Auto-Grow vs Traditional Fixed Grid',
    fontsize=14, fontweight='bold', y=1.01,
)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), '..', 'ab_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存至: {output_path}")

# ==========================================================================
# 汇总
# ==========================================================================
print("\n" + "=" * 78)
print(f"{'指标':<28} {'AdaGrid':>16} {'Trad G=3':>16} {'Trad G=50':>16}")
print("-" * 78)
print(f"{'最终 MSE':<28} {ada_mse:>16.6f} {trad_results[3]['mse']:>16.6f} {trad_results[50]['mse']:>16.6f}")
print(f"{'最终 Grid':<28} {ada_model.grid_size:>16} {3:>16} {50:>16}")
print(f"{'翻倍/固定':<28} {len(ada_events):>16} {'fixed':>16} {'fixed':>16}")

imp_vs_g3 = (trad_results[3]['mse'] - ada_mse) / trad_results[3]['mse'] * 100
imp_vs_g50 = (trad_results[50]['mse'] - ada_mse) / trad_results[50]['mse'] * 100
print(f"\n★ AdaGrid vs 传统 G=3:  MSE {'领先' if imp_vs_g3 > 0 else '落后'} {abs(imp_vs_g3):.1f}%")
print(f"★ AdaGrid vs 传统 G=50: MSE {'领先' if imp_vs_g50 > 0 else '落后'} {abs(imp_vs_g50):.1f}%")
print("  （同 G 同 epoch 时传统 G=50 可能更好——AdaGrid 价值在于无需事先猜 G）")
print("=" * 78)
