# Quick Start: AdaGrid-KAN / EcoGrow-KAN

This guide shows how to install dependencies, run the main experiments, open the local dashboard, and inspect generated results.

## 1. Prepare Python

Use Python 3.10 or newer.

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, you can still run commands through the virtual environment directly:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe examples\ecogrow_comparison.py
```

## 2. Run the Main Comparison

Run Fixed KAN, AdaGrid-KAN, and EcoGrow-KAN on the synthetic one-dimensional regression task:

```bash
python examples/ecogrow_comparison.py
```

The script trains three models:

- `Fixed KAN`: fixed grid size for the whole run.
- `AdaGrid-KAN`: expands the grid whenever training loss plateaus.
- `EcoGrow-KAN`: tries expansion, keeps it only when validation improvement justifies the extra parameters, and rolls back weak growth.

Expected outputs in the repository root:

- `ecogrow_comparison.png`
- `ecogrow_training_history.png`

## 3. Run a Faster Smoke Test

For a quick check that the code path works:

```bash
python examples/ecogrow_comparison.py --epochs 80 --trial-epochs 10
```

This is useful for verifying installation, but it is too short to judge the algorithm.

## 4. Run Multi-Seed Results

To reproduce the README-style multi-seed report:

```bash
python examples/ecogrow_comparison.py --seeds 42 43 44 45 46
```

This takes longer because each seed trains Fixed KAN, AdaGrid-KAN, and EcoGrow-KAN.

## 5. Run the Basic Examples

Basic AdaGrid demo:

```bash
python examples/toy_regression.py
```

Output:

- `regression_result.png`

Fixed-grid vs AdaGrid A/B comparison:

```bash
python examples/ab_comparison.py
```

Output:

- `ab_comparison.png`

## 6. Start the Local Dashboard

The dashboard provides a browser UI for launching runs and viewing metrics.

Start the FastAPI server:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The dashboard uses these local API endpoints:

- `GET /api/health`
- `POST /api/runs`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/stream`

Training runs are saved under:

```text
runs/<run_id>/
```

Each run directory contains:

- `config.json`: normalized run configuration.
- `metrics.jsonl`: per-epoch training metrics.
- `events.json`: grid-growth decisions.
- `summary.json`: final MSE, grid size, parameter count, and status.
- `fit_data.json`: data needed to redraw the final fit.

## 7. Run Tests

```bash
pytest -q
```

The test suite checks:

- EcoGrow trial growth.
- Accept/reject decisions.
- Rollback behavior.
- Rejection memory.
- Maximum grid bounds.
- Invalid loss handling.
- Training runner persistence.
- Dashboard API behavior.

## 8. Common Parameters

For `examples/ecogrow_comparison.py`:

```bash
python examples/ecogrow_comparison.py --help
```

Frequently useful options:

```bash
python examples/ecogrow_comparison.py --epochs 1200
python examples/ecogrow_comparison.py --noise-std 0.15
python examples/ecogrow_comparison.py --initial-grid 3 --max-grid 50
python examples/ecogrow_comparison.py --patience 8
python examples/ecogrow_comparison.py --trial-epochs 30
python examples/ecogrow_comparison.py --min-improvement 0.01 --min-efficiency 0.05
```

Important meanings:

- `initial-grid`: starting grid for AdaGrid and EcoGrow.
- `fixed-grid`: grid used by Fixed KAN.
- `max-grid`: upper bound for automatic expansion.
- `patience`: plateau epochs before trying expansion.
- `trial-epochs`: how long EcoGrow trains after trial expansion before deciding.
- `min-improvement`: minimum relative validation improvement required.
- `min-efficiency`: minimum validation-improvement-per-parameter-growth score required.

## 9. What to Look For

After a run, compare:

- Final grid size.
- Trainable parameter count.
- Validation MSE.
- Test MSE.
- EcoGrow accepted/rejected/blocked events.

Typical behavior:

- Fixed KAN is simple but depends on choosing a good grid before training.
- AdaGrid usually grows aggressively once loss plateaus.
- EcoGrow may stop earlier because it can roll back expansions with weak validation payoff.

This project is an algorithm demonstration. The provided synthetic regression task is useful for understanding the mechanism, but broader performance claims require more datasets and baselines.
