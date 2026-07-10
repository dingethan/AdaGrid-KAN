#!/usr/bin/env python3
"""
AdaGrid-KAN 1D 回归演示（均匀 Grid 自动生长）
=============================================
目标函数: y = sin(5x) + 0.5 * cos(15x)

策略（与 visualization/index.html 一致）:
  G=3 起步 → loss plateau → 均匀翻倍 → 最长至 G=50
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

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 1200
INITIAL_GRID = 3
MAX_GRID = 50
PATIENCE = 8
LR = 0.03
MIN_DELTA_ABS = 1e-8
MIN_DELTA_REL = 0.005

print(f"Device: {DEVICE}")
print(f"AdaGrid: G={INITIAL_GRID} → {MAX_GRID}, Patience={PATIENCE}")


def target_function(x):
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


x_train = torch.linspace(-1, 1, 200).unsqueeze(1)
y_train = target_function(x_train)
x_test = torch.linspace(-1, 1, 500).unsqueeze(1)
y_test = target_function(x_test)

model = DynamicKANLayer(
    in_dim=1, out_dim=1, grid_size=INITIAL_GRID,
    spline_order=3, grid_range=(-1.0, 1.0),
).to(DEVICE)

scheduler = ExtendGridOnPlateau(
    model, patience=PATIENCE, max_grid=MAX_GRID,
    verbose=True, min_delta=MIN_DELTA_ABS, min_delta_rel=MIN_DELTA_REL,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

x_train_d = x_train.to(DEVICE)
y_train_d = y_train.to(DEVICE)

loss_history = []
refine_epochs = []
grid_history = []

print("\n" + "=" * 60)
print("开始训练（均匀 Grid 自动翻倍）...")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    pred = model(x_train_d)
    loss = criterion(pred, y_train_d)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)
    grid_history.append(model.grid_size)

    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | Loss: {loss_val:.6f} | Grid: {model.grid_size}")

    refined, events = scheduler.step(loss_val, epoch=epoch)
    if refined:
        for ep, old_g, new_g in events:
            refine_epochs.append(ep)
            print(f"  >> Uniform double: G {old_g} → {new_g} at epoch {ep}")
        current_lr = optimizer.param_groups[0]['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-5)

print("\n" + "=" * 60)
print(f"训练完成！Loss={loss_history[-1]:.6f}, Grid={model.grid_size}, 翻倍 {len(refine_epochs)} 次")
print("=" * 60)

model.eval()
with torch.no_grad():
    y_pred = model(x_test.to(DEVICE)).cpu()

x_test_np = x_test.cpu().numpy().flatten()
y_test_np = y_test.cpu().numpy().flatten()
y_pred_np = y_pred.cpu().numpy().flatten()
final_mse = np.mean((y_pred_np - y_test_np) ** 2)
print(f"\n最终测试 MSE: {final_mse:.6f}")

# ------------------------------------------------------------------
# 三子图：Loss / 拟合 / Grid 生长
# ------------------------------------------------------------------
plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax1, ax2, ax3 = axes

ax1.plot(range(1, NUM_EPOCHS + 1), loss_history, color='#ff7a1a', linewidth=1.0)
ax1.set_yscale('log')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE, log)')
ax1.set_title('Loss + Uniform Grid Doubling')
ax1.grid(True, alpha=0.3)
for ep in refine_epochs:
    ax1.axvline(x=ep, color='#ff7a1a', linestyle='--', linewidth=1.0, alpha=0.6)
if refine_epochs:
    ax1.axvline(x=refine_epochs[0], color='#ff7a1a', linestyle='--', linewidth=1.0,
                alpha=0.6, label=f'Grid double ({len(refine_epochs)}×)')
    ax1.legend(fontsize=8)

ax2.plot(x_test_np, y_test_np, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.7)
ax2.plot(x_test_np, y_pred_np, color='#ff7a1a', linewidth=1.8,
         label=f'AdaGrid (G={model.grid_size})')
grid_np = model.grid.cpu().numpy()
for x in grid_np:
    ax2.axvline(x=x, color='#ff7a1a', alpha=0.3, linewidth=0.8, linestyle='--')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'MSE={final_mse:.6f} | Uniform Grid')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3.plot(range(1, NUM_EPOCHS + 1), grid_history, color='#ff7a1a', linewidth=1.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Grid G')
ax3.set_title('Grid Auto-Grow (3→6→12→24→48→50)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), '..')
output_path = os.path.join(output_dir, 'regression_result.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")
