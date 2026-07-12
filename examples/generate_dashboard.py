"""
综合对比仪表盘 - 数据收集与 HTML 生成
==========================================
运行所有核心实验并生成一个交互式 HTML 仪表盘，汇总所有结果。
"""

from __future__ import annotations

import json
import os
import sys

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


# ==========================================================================
# 快速收集实验数据
# ==========================================================================

def target_function(x):
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


def collect_dashboard_data(seed=42, epochs=1200, noise_std=0.15):
    """收集仪表盘需要的所有实验数据。"""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train, n_val, n_test = 160, 160, 400

    # 数据
    x_train = torch.linspace(-1, 1, n_train).unsqueeze(1)
    x_val = torch.linspace(-1, 1, n_val).unsqueeze(1)
    x_test = torch.linspace(-1, 1, n_test).unsqueeze(1)
    y_train = target_function(x_train) + noise_std * torch.randn_like(target_function(x_train))
    y_val = target_function(x_val)
    y_test = target_function(x_test)

    x_train_d, y_train_d = x_train.to(device), y_train.to(device)
    x_val_d, y_val_d = x_val.to(device), y_val.to(device)
    criterion = nn.MSELoss()

    data = {}

    # ---- Fixed KAN ----
    torch.manual_seed(seed)
    model_f = AdaptiveKANLayer(1, 1, grid_size=12, spline_order=3).to(device)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=0.03, weight_decay=1e-5)
    f_loss, f_val = [], []
    for epoch in range(epochs):
        model_f.train()
        pred = model_f(x_train_d)
        loss = criterion(pred, y_train_d)
        opt_f.zero_grad(); loss.backward(); opt_f.step()
        f_loss.append(loss.item())
        model_f.eval()
        with torch.no_grad():
            f_val.append(criterion(model_f(x_val_d), y_val_d).item())
    data["fixed"] = {"train_loss": f_loss, "val_loss": f_val, "params": count_trainable_parameters(model_f)}

    # ---- AdaGrid-KAN ----
    torch.manual_seed(seed)
    model_a = AdaptiveKANLayer(1, 1, grid_size=3, spline_order=3).to(device)
    sched_a = PlateauGridExpander(model_a, patience=10, max_grid=64, verbose=False)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=0.03, weight_decay=1e-5)
    a_loss, a_val, a_grid = [], [], []
    a_events = []
    for epoch in range(epochs):
        model_a.train()
        pred = model_a(x_train_d)
        loss = criterion(pred, y_train_d)
        opt_a.zero_grad(); loss.backward(); opt_a.step()
        a_loss.append(loss.item())
        model_a.eval()
        with torch.no_grad():
            a_val.append(criterion(model_a(x_val_d), y_val_d).item())
        a_grid.append(model_a.grid_size)
        expanded, events = sched_a.step(loss.item(), epoch=epoch)
        if expanded:
            a_events.extend(events)
            opt_a = torch.optim.Adam(model_a.parameters(), lr=0.03, weight_decay=1e-5)
    data["adagrid"] = {
        "train_loss": a_loss, "val_loss": a_val, "grid": a_grid,
        "events": [(int(e[0]), int(e[1]), int(e[2])) for e in a_events],
        "params": count_trainable_parameters(model_a),
    }

    # ---- EcoGrow-KAN ----
    torch.manual_seed(seed)
    model_e = AdaptiveKANLayer(1, 1, grid_size=3, spline_order=3).to(device)
    sched_e = EcoGrowScheduler(
        model_e, patience=10, max_grid=64,
        trial_epochs=25, min_improvement=0.01, min_efficiency=0.05, verbose=False,
    )
    opt_e = torch.optim.Adam(model_e.parameters(), lr=0.03, weight_decay=1e-5)
    e_loss, e_val, e_grid = [], [], []
    e_events = []
    for epoch in range(epochs):
        model_e.train()
        pred = model_e(x_train_d)
        loss = criterion(pred, y_train_d)
        opt_e.zero_grad(); loss.backward(); opt_e.step()
        e_loss.append(loss.item())
        model_e.eval()
        with torch.no_grad():
            val_l = criterion(model_e(x_val_d), y_val_d).item()
        e_val.append(val_l)
        e_grid.append(model_e.grid_size)
        result = sched_e.step(train_loss=loss.item(), val_loss=val_l, epoch=epoch)
        if result.optimizer_reset_required:
            opt_e = torch.optim.Adam(model_e.parameters(), lr=0.03, weight_decay=1e-5)
    for ev in sched_e.events:
        e_events.append({
            "epoch": int(ev["epoch"]),
            "old_grid": int(ev["old_grid"]),
            "new_grid": int(ev["new_grid"]),
            "decision": ev["decision"],
            "relative_improvement": float(ev["relative_improvement"]),
            "efficiency_score": float(ev["efficiency_score"]),
        })
    data["ecogrow"] = {
        "train_loss": e_loss, "val_loss": e_val, "grid": e_grid,
        "events": e_events, "params": count_trainable_parameters(model_e),
        "accepted": sched_e.num_accepted,
        "rejected": sched_e.num_rejected,
        "blocked": sched_e.num_blocked,
    }

    # ---- Fitting curves ----
    model_f.eval()
    model_a.eval()
    model_e.eval()
    with torch.no_grad():
        y_fixed = model_f(x_test.to(device)).cpu().numpy().flatten()
        y_ada = model_a(x_test.to(device)).cpu().numpy().flatten()
        y_eco = model_e(x_test.to(device)).cpu().numpy().flatten()
    data["fitting"] = {
        "x": x_test.numpy().flatten().tolist(),
        "y_true": y_test.numpy().flatten().tolist(),
        "y_fixed": y_fixed.tolist(),
        "y_ada": y_ada.tolist(),
        "y_eco": y_eco.tolist(),
    }

    data["meta"] = {
        "epochs": epochs,
        "noise_std": noise_std,
        "seed": seed,
    }

    return data


# ==========================================================================
# HTML 生成
# ==========================================================================

def generate_dashboard_html(data, output_path):
    """生成交互式 HTML 仪表盘。"""
    meta = data["meta"]

    # 将数据序列化为 JSON
    json_data = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdaGrid-KAN 综合实验仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }}
  .header {{ text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e, #16213e); }}
  .header h1 {{ font-size: 28px; color: #ff7a1a; }}
  .header p {{ color: #888; margin-top: 8px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{ background: #1a1a2e; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
  .card h2 {{ font-size: 16px; color: #ff7a1a; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .metric-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
  .metric {{ background: #1a1a2e; border-radius: 12px; padding: 20px; flex: 1; min-width: 200px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
  .metric .value {{ font-size: 32px; font-weight: bold; margin: 8px 0; }}
  .metric .label {{ color: #888; font-size: 13px; }}
  .fixed-color {{ color: #4a6cf7; }}
  .ada-color {{ color: #ff7a1a; }}
  .eco-color {{ color: #20b38a; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ padding: 10px 14px; text-align: center; border-bottom: 1px solid #333; }}
  th {{ color: #ff7a1a; font-weight: 600; }}
  .event-table td {{ font-size: 13px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .badge-accepted {{ background: #20b38a33; color: #20b38a; }}
  .badge-rejected {{ background: #e0455b33; color: #e0455b; }}
  .badge-blocked {{ background: #88888833; color: #888; }}
  canvas {{ max-height: 350px; }}
</style>
</head>
<body>
<div class="header">
  <h1>AdaGrid-KAN / EcoGrow-KAN 综合实验仪表盘</h1>
  <p>Epochs: {meta["epochs"]} | Noise: {meta["noise_std"]} | Seed: {meta["seed"]}</p>
</div>

<div class="container">
  <!-- 关键指标 -->
  <div class="metric-row">
    <div class="metric">
      <div class="label">Fixed KAN 参数</div>
      <div class="value fixed-color">{data["fixed"]["params"]}</div>
      <div class="label">Grid = 12</div>
    </div>
    <div class="metric">
      <div class="label">AdaGrid-KAN 参数</div>
      <div class="value ada-color">{data["adagrid"]["params"]}</div>
      <div class="label">Final Grid = {data["adagrid"]["grid"][-1]}</div>
    </div>
    <div class="metric">
      <div class="label">EcoGrow-KAN 参数</div>
      <div class="value eco-color">{data["ecogrow"]["params"]}</div>
      <div class="label">Final Grid = {data["ecogrow"]["grid"][-1]}</div>
    </div>
    <div class="metric">
      <div class="label">EcoGrow 接受/拒绝/阻止</div>
      <div class="value eco-color">{data["ecogrow"]["accepted"]}/{data["ecogrow"]["rejected"]}/{data["ecogrow"]["blocked"]}</div>
      <div class="label">扩容决策</div>
    </div>
  </div>

  <!-- 图表 -->
  <div class="grid">
    <div class="card">
      <h2>训练 Loss 对比</h2>
      <canvas id="trainLossChart"></canvas>
    </div>
    <div class="card">
      <h2>验证 Loss 对比</h2>
      <canvas id="valLossChart"></canvas>
    </div>
    <div class="card">
      <h2>Grid Size 变化</h2>
      <canvas id="gridChart"></canvas>
    </div>
    <div class="card">
      <h2>拟合曲线对比</h2>
      <canvas id="fittingChart"></canvas>
    </div>
  </div>

  <!-- EcoGrow 事件表 -->
  <div class="card full-width">
    <h2>EcoGrow 扩容决策事件</h2>
    <table class="event-table">
      <tr>
        <th>Epoch</th><th>Grid 变化</th><th>决策</th>
        <th>验证改善率</th><th>效率分数</th>
      </tr>
      {"".join(
        f'<tr><td>{e["epoch"]}</td><td>{e["old_grid"]} → {e["new_grid"]}</td>'
        f'<td><span class="badge badge-{e["decision"]}">{e["decision"]}</span></td>'
        f'<td>{e["relative_improvement"]*100:.2f}%</td>'
        f'<td>{e["efficiency_score"]:.3f}</td></tr>'
        for e in data["ecogrow"]["events"]
      )}
    </table>
  </div>
</div>

<script>
const DATA = {json_data};

// 通用配置
const chartDefaults = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#ccc', font: {{ size: 12 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
    y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
  }}
}};

const epochs = Array.from({{ length: DATA.meta.epochs }}, (_, i) => i + 1);

// 训练 Loss
new Chart(document.getElementById('trainLossChart'), {{
  type: 'line',
  data: {{
    labels: epochs,
    datasets: [
      {{ label: 'Fixed KAN', data: DATA.fixed.train_loss, borderColor: '#4a6cf7', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'AdaGrid-KAN', data: DATA.adagrid.train_loss, borderColor: '#ff7a1a', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'EcoGrow-KAN', data: DATA.ecogrow.train_loss, borderColor: '#20b38a', borderWidth: 1.5, pointRadius: 0 }},
    ]
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, type: 'logarithmic' }} }} }}
}});

// 验证 Loss
new Chart(document.getElementById('valLossChart'), {{
  type: 'line',
  data: {{
    labels: epochs,
    datasets: [
      {{ label: 'Fixed KAN', data: DATA.fixed.val_loss, borderColor: '#4a6cf7', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'AdaGrid-KAN', data: DATA.adagrid.val_loss, borderColor: '#ff7a1a', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'EcoGrow-KAN', data: DATA.ecogrow.val_loss, borderColor: '#20b38a', borderWidth: 1.5, pointRadius: 0 }},
    ]
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, type: 'logarithmic' }} }} }}
}});

// Grid Size
new Chart(document.getElementById('gridChart'), {{
  type: 'line',
  data: {{
    labels: epochs,
    datasets: [
      {{ label: 'AdaGrid Grid', data: DATA.adagrid.grid, borderColor: '#ff7a1a', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'EcoGrow Grid', data: DATA.ecogrow.grid, borderColor: '#20b38a', borderWidth: 2, pointRadius: 0 }},
    ]
  }},
  options: chartDefaults
}});

// 拟合曲线
new Chart(document.getElementById('fittingChart'), {{
  type: 'line',
  data: {{
    labels: DATA.fitting.x,
    datasets: [
      {{ label: 'Ground Truth', data: DATA.fitting.y_true, borderColor: '#ffffff', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'Fixed KAN', data: DATA.fitting.y_fixed, borderColor: '#4a6cf7', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'AdaGrid-KAN', data: DATA.fitting.y_ada, borderColor: '#ff7a1a', borderWidth: 1.5, pointRadius: 0 }},
      {{ label: 'EcoGrow-KAN', data: DATA.fitting.y_eco, borderColor: '#20b38a', borderWidth: 1.5, pointRadius: 0 }},
    ]
  }},
  options: chartDefaults
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"仪表盘已保存至: {output_path}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..")
    html_path = os.path.join(output_dir, "dashboard.html")

    print("收集实验数据...")
    data = collect_dashboard_data(seed=42, epochs=1200, noise_std=0.15)

    print("生成仪表盘 HTML...")
    generate_dashboard_html(data, html_path)

    print("完成！请在浏览器中打开 dashboard.html 查看。")


if __name__ == "__main__":
    main()
