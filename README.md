# AdaGrid-KAN with EcoGrow

Cost-aware reversible grid expansion for KAN.

This repository is a small PyTorch research/demo project around **Kolmogorov-Arnold Networks (KAN)**. It implements a dynamic B-spline KAN layer and compares three grid strategies for one-dimensional regression:

- **Fixed KAN**: choose the grid size before training and keep it unchanged.
- **AdaGrid-KAN**: start with a small grid and expand it when training loss reaches a plateau.
- **EcoGrow-KAN**: try the same expansion, evaluate validation improvement against parameter cost, keep useful growth, roll back weak growth, and remember rejected transitions.

For a runnable checklist, see [docs/quick_start.md](docs/quick_start.md).

## What This Project Does

KAN layers use spline grids to represent nonlinear functions. A grid that is too small can underfit; a grid that is too large wastes parameters and can overfit noisy data. This project explores a middle path:

```text
start with a coarse grid
train normally
detect a training-loss plateau
try a larger grid
measure validation gain and parameter growth
accept the expansion or roll back
```

The main experiment fits the synthetic target function:

```python
y = sin(5x) + 0.5 * cos(15x)
```

Training data can include Gaussian noise, while validation and test data use the clean function. The project records loss curves, grid growth history, expansion decisions, parameter counts, and final fit data.

## Core Idea

AdaGrid asks:

> When should the grid grow?

EcoGrow adds:

> Was this growth useful enough to justify the extra trainable parameters?

EcoGrow performs trial expansion and computes:

```python
relative_improvement = (val_loss_before - best_trial_val_loss) / val_loss_before
relative_param_growth = (params_after - params_before) / params_before
efficiency_score = relative_improvement / relative_param_growth
```

Growth is accepted only when both thresholds are met:

- `relative_improvement >= min_improvement`
- `efficiency_score >= min_efficiency`

Otherwise the scheduler restores the model backup and blocks the same rejected `(old_grid -> new_grid)` transition from being retried.

## Repository Layout

```text
AdaGrid-KAN/
+-- src/
|   +-- kan_layer.py              # DynamicKANLayer and B-spline grid refinement
|   +-- grid_scheduler.py         # AdaGrid and EcoGrow schedulers
|   +-- training/
|       +-- datasets.py           # Synthetic regression data
|       +-- persistence.py        # JSON/JSONL run artifacts
|       +-- runner.py             # Shared training runner for dashboard/API
+-- backend/
|   +-- app.py                    # FastAPI app for local dashboard runs
|   +-- run_manager.py            # Background run lifecycle and SSE streaming
|   +-- schemas.py                # Run config validation wrapper
+-- examples/
|   +-- toy_regression.py         # Basic AdaGrid regression demo
|   +-- ab_comparison.py          # Fixed KAN vs AdaGrid-KAN
|   +-- ecogrow_comparison.py     # Fixed KAN vs AdaGrid-KAN vs EcoGrow-KAN
+-- visualization/
|   +-- ecogrow_live_comparison.html
+-- tests/
|   +-- test_ecogrow_scheduler.py
|   +-- test_backend_api.py
|   +-- test_training_dashboard.py
+-- docs/
|   +-- quick_start.md
|   +-- images/
+-- requirements.txt
```

## Quick Start

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the main comparison:

```bash
python examples/ecogrow_comparison.py
```

Run a multi-seed comparison:

```bash
python examples/ecogrow_comparison.py --seeds 42 43 44 45 46
```

Run tests:

```bash
pytest -q
```

Start the local dashboard/API:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

The dashboard starts training runs, streams metrics, and reads saved run artifacts from `runs/`.

## Main Components

### DynamicKANLayer

Defined in `src/kan_layer.py`.

`DynamicKANLayer` implements a compact KAN layer for regression:

- Builds a clamped B-spline knot vector from the current grid.
- Computes B-spline basis values with Cox-de Boor recursion.
- Combines a SiLU base term with spline control-point weights.
- Supports global grid refinement through `refine_grid(new_grid_size)`.

`refine_grid()` increases the number of grid intervals and migrates existing control points with `torch.nn.functional.interpolate`, so training can continue after expansion instead of restarting from scratch.

The file also contains `insert_knot()`, a local Boehm knot-insertion method. It is kept as an advanced interface, but the current examples and schedulers primarily use global uniform refinement.

### ExtendGridOnPlateau

Defined in `src/grid_scheduler.py`.

This is the AdaGrid scheduler. It watches training loss and expands every `DynamicKANLayer` when loss fails to improve for `patience` epochs:

```text
G -> min(G * growth_factor, max_grid)
```

The default demo trajectory can look like:

```text
3 -> 6 -> 12 -> 24 -> 48 -> 50
```

### EcoGrowScheduler

Defined in `src/grid_scheduler.py`.

EcoGrow extends AdaGrid with a reversible trial phase:

1. Save a deep copy of the current model.
2. Expand the grid.
3. Train for `trial_epochs`.
4. Track the best validation loss during the trial.
5. Accept the expansion if validation improvement is efficient enough.
6. Otherwise restore the backup model and remember the rejected transition.

The scheduler returns an `EcoGrowResult`. Training loops must use `result.model` after every step because rejection can replace the model with a restored backup. Optimizers must also be rebuilt when `optimizer_reset_required` is true.

## Example API

```python
import torch
import torch.nn.functional as F

from src.grid_scheduler import EcoGrowScheduler
from src.kan_layer import DynamicKANLayer

model = DynamicKANLayer(in_dim=1, out_dim=1, grid_size=3, spline_order=3)
scheduler = EcoGrowScheduler(
    model,
    patience=8,
    max_grid=50,
    trial_epochs=30,
    min_improvement=0.01,
    min_efficiency=0.05,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-5)

for epoch in range(num_epochs):
    model.train()
    train_loss = F.mse_loss(model(x_train), y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = F.mse_loss(model(x_val), y_val)

    result = scheduler.step(
        train_loss=train_loss.item(),
        val_loss=val_loss.item(),
        epoch=epoch,
    )
    model = result.model

    if result.optimizer_reset_required:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-5)
```

## Experiment Scripts

`examples/toy_regression.py`

- Trains AdaGrid-KAN on the clean synthetic function.
- Saves `regression_result.png`.

`examples/ab_comparison.py`

- Compares AdaGrid-KAN against fixed-grid KAN baselines.
- Saves `ab_comparison.png`.

`examples/ecogrow_comparison.py`

- Compares Fixed KAN, AdaGrid-KAN, and EcoGrow-KAN.
- Supports single-seed and multi-seed runs.
- Saves `ecogrow_comparison.png` and `ecogrow_training_history.png`.

Useful command-line options:

```bash
python examples/ecogrow_comparison.py --epochs 1200 --noise-std 0.15
python examples/ecogrow_comparison.py --trial-epochs 30 --min-improvement 0.01 --min-efficiency 0.05
python examples/ecogrow_comparison.py --seeds 42 43 44 45 46
```

## Dashboard/API

The local dashboard is served by FastAPI from `backend/app.py` and uses `visualization/ecogrow_live_comparison.html` as the UI.

Start it with:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET /`: dashboard page.
- `GET /api/health`: checks Python, Torch, imports, and run-directory writability.
- `POST /api/runs`: starts one training run.
- `GET /api/runs`: lists saved runs.
- `GET /api/runs/{run_id}`: reads one saved run.
- `GET /api/runs/{run_id}/stream`: streams live metrics/events with Server-Sent Events.

Each run writes artifacts under `runs/<run_id>/`:

- `config.json`
- `metrics.jsonl`
- `events.json`
- `summary.json`
- `fit_data.json`

## Results

The figures below are representative outputs, not universal performance guarantees.

![Fitting and validation-loss comparison](docs/images/ecogrow_comparison.png)

![Grid history with EcoGrow decisions](docs/images/ecogrow_training_history.png)

In the README demonstration configuration, EcoGrow often ends with fewer trainable parameters than AdaGrid because it can reject late expansions whose validation benefit is too small. Validation performance is seed-dependent: parameter savings are relatively stable in the provided experiments, while validation MSE improvement varies across seeds.

## Tests

Run:

```bash
pytest -q
```

The tests cover:

- EcoGrow growth start.
- Accepting useful growth.
- Rejecting and rolling back weak growth.
- Maximum grid bounds.
- Cooldown behavior.
- Rejection memory.
- Invalid loss handling.
- Training-run persistence and dashboard API behavior.

## Design Limits

- Experiments focus on one-dimensional synthetic regression.
- The model and plots are meant for algorithm demonstration, not state-of-the-art benchmarking.
- Trial growth costs extra training epochs.
- Optimizers must be rebuilt when parameter shapes change.
- Thresholds such as `patience`, `trial_epochs`, `min_improvement`, and `min_efficiency` materially affect outcomes.
- Broader claims would require more datasets, architectures, and baselines.

## License

MIT.
