# KAN Training Dashboard Design

## Goal

Build a local experiment dashboard for comparing KAN grid strategies. The existing `visualization/ecogrow_live_comparison.html` will be converted from an in-browser JavaScript simulation into a frontend that controls and visualizes real Python/PyTorch training through a FastAPI backend.

The tool is local-only. It is intended for project demonstration and repeated experiments on one machine, not multi-user deployment.

## User Flow

1. User starts the FastAPI backend.
2. User opens the dashboard at `http://127.0.0.1:8000/`.
3. The page calls `/api/health` and displays backend, Python, Torch, CUDA, module import, and run-directory status.
4. If the environment is healthy, the user selects one training strategy:
   - `fixed`
   - `adagrid`
   - `ecogrow`
5. User configures training parameters such as epochs, seed, noise, grid sizes, patience, learning rate, and EcoGrow thresholds.
6. User starts a run.
7. Backend trains with existing Python code and streams metrics to the frontend.
8. Frontend displays live train loss, validation loss, grid size, parameter count, current status, and grid-strategy events.
9. On completion, backend saves the run under `runs/<run_id>/`.
10. Frontend refreshes history and lets the user select which saved runs to show in comparison charts.

## Architecture

### Backend

Add a FastAPI backend under `backend/`.

Files:

- `backend/app.py`: FastAPI app, static frontend serving, API routes.
- `backend/run_manager.py`: in-memory active-run registry, background task startup, status tracking.
- `backend/schemas.py`: request/response models or typed dict-style payload definitions.

The backend owns:

- Environment health checks.
- Starting training tasks.
- Streaming live metrics with Server-Sent Events.
- Listing and reading saved runs.

### Training Layer

Add reusable training code under `src/training/`.

Files:

- `src/training/datasets.py`: deterministic 1D synthetic dataset generation.
- `src/training/runner.py`: unified runners for Fixed KAN, AdaGrid-KAN, and EcoGrow-KAN.
- `src/training/persistence.py`: run directory creation and JSON/JSONL artifact writing.

The training layer must reuse:

- `src.kan_layer.DynamicKANLayer`
- `src.grid_scheduler.ExtendGridOnPlateau`
- `src.grid_scheduler.EcoGrowScheduler`
- `src.grid_scheduler.count_trainable_parameters`

It should not depend on frontend code.

### Frontend

Refactor the existing file:

- `visualization/ecogrow_live_comparison.html`

The page should keep the existing dashboard idea but stop doing JavaScript training. It should remove or stop using the in-browser `KAN1D`, `AdaScheduler`, and `EcoScheduler` simulation code.

The frontend owns:

- Health check display.
- Run configuration form.
- Run start button.
- SSE connection for live metrics.
- History list with checkboxes.
- Charts for selected runs.
- Event log display.

## API

### `GET /`

Returns `visualization/ecogrow_live_comparison.html`.

### `GET /api/health`

Returns environment status:

```json
{
  "ok": true,
  "python": "3.x",
  "torch": {
    "available": true,
    "version": "2.x",
    "cuda_available": false,
    "device": "cpu"
  },
  "modules": {
    "DynamicKANLayer": true,
    "schedulers": true
  },
  "runs_dir_writable": true,
  "errors": []
}
```

If any required part is unavailable, `ok` is false and `errors` explains why.

### `POST /api/runs`

Starts one training run.

Request:

```json
{
  "strategy": "ecogrow",
  "epochs": 700,
  "seed": 42,
  "noise_std": 0.15,
  "train_size": 160,
  "val_size": 160,
  "test_size": 400,
  "initial_grid": 3,
  "fixed_grid": 12,
  "max_grid": 50,
  "patience": 8,
  "lr": 0.03,
  "weight_decay": 0.00001,
  "trial_epochs": 30,
  "min_improvement": 0.01,
  "min_efficiency": 0.05
}
```

Response:

```json
{
  "run_id": "20260712-141530-ecogrow-seed42",
  "status": "running"
}
```

### `GET /api/runs/{run_id}/stream`

Streams Server-Sent Events for one active run.

Event types:

- `metric`: epoch-level metric payload.
- `event`: grid strategy event payload.
- `completed`: final summary payload.
- `failed`: error payload.

Metric payload:

```json
{
  "run_id": "20260712-141530-ecogrow-seed42",
  "epoch": 120,
  "train_loss": 0.031,
  "val_loss": 0.008,
  "grid": 12,
  "params": 16
}
```

Event payload:

```json
{
  "run_id": "20260712-141530-ecogrow-seed42",
  "epoch": 150,
  "type": "accepted",
  "old_grid": 6,
  "new_grid": 12,
  "relative_improvement": 0.034,
  "relative_param_growth": 0.5,
  "efficiency_score": 0.068
}
```

### `GET /api/runs`

Lists saved runs by reading `runs/*/summary.json` and `runs/*/config.json`.

### `GET /api/runs/{run_id}`

Returns a full saved run:

```json
{
  "config": {},
  "summary": {},
  "metrics": [],
  "events": [],
  "fit_data": {}
}
```

## Run Storage

Each run is saved under:

```text
runs/<run_id>/
  config.json
  metrics.jsonl
  events.json
  summary.json
  fit_data.json
```

`metrics.jsonl` is appended during training, one JSON object per epoch.

`fit_data.json` stores enough points for frontend rendering:

```json
{
  "x_train": [],
  "y_train": [],
  "x_test": [],
  "y_test": [],
  "y_pred": []
}
```

## Frontend Charts

The dashboard should show selected saved runs, not only the current run.

Charts:

- Train loss vs epoch.
- Validation loss vs epoch.
- Grid size vs epoch.
- Final fit comparison with true curve, noisy training points, and selected run predictions.

The run history list uses checkboxes. A checked run appears on charts; an unchecked run is hidden. The current run is checked automatically after it starts or completes.

## Error Handling

- If `/api/health` fails, disable the start button and show the error.
- If a run fails, stream a `failed` event, write an error summary if possible, and show the message in the frontend.
- If a saved run is malformed, skip it in the run list and show a non-blocking warning.
- The first version supports one active run at a time. If a run is already active, `POST /api/runs` returns an error.

## Scope

Included in first implementation:

- Local FastAPI backend.
- Reuse existing HTML file as frontend.
- Health check.
- Single-strategy runs.
- Live metric streaming through SSE.
- Saved run artifacts.
- History list and checkbox-based comparison.

Excluded from first implementation:

- Multi-user support.
- Authentication.
- Remote deployment.
- Concurrent training runs.
- Database storage.
- Hard cancellation of running training.
- Exporting comparison charts as image files.
