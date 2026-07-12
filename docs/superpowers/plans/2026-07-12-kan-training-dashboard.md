# KAN Training Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI-backed dashboard that runs real KAN training strategies and visualizes live and saved experiment results.

**Architecture:** Add a reusable Python training layer that streams epoch metrics and persists run artifacts. Add a FastAPI backend that serves the existing HTML frontend, starts one background run at a time, exposes health/history/detail APIs, and streams metrics through SSE. Refactor `visualization/ecogrow_live_comparison.html` so it no longer trains in JavaScript and instead renders backend-provided runs.

**Tech Stack:** Python, PyTorch, FastAPI, Server-Sent Events, plain HTML/CSS/JavaScript, SVG charts.

---

## File Structure

- Create `backend/__init__.py`: package marker.
- Create `backend/app.py`: FastAPI app, health API, run APIs, static HTML route.
- Create `backend/run_manager.py`: active run state, worker thread, queue-backed event stream, saved-run loading.
- Create `backend/schemas.py`: request defaults and validation helpers.
- Create `src/training/__init__.py`: package marker.
- Create `src/training/datasets.py`: deterministic synthetic dataset generation.
- Create `src/training/persistence.py`: JSON/JSONL artifact helpers.
- Create `src/training/runner.py`: real Fixed/AdaGrid/EcoGrow training loop with metric callback.
- Modify `visualization/ecogrow_live_comparison.html`: replace in-browser simulation with API-backed dashboard.
- Modify `requirements.txt`: add FastAPI and Uvicorn.
- Create `tests/test_training_dashboard.py`: training runner and persistence tests.
- Create `tests/test_backend_api.py`: FastAPI route tests.

## Task 1: Training Dataset And Persistence

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/datasets.py`
- Create: `src/training/persistence.py`
- Test: `tests/test_training_dashboard.py`

- [ ] **Step 1: Write tests for dataset and persistence**

Add tests that assert deterministic dataset generation and run artifact writing.

- [ ] **Step 2: Implement dataset generation**

Create a helper that returns train, val, and test tensors for `sin(5x) + 0.5*cos(15x)` with deterministic noisy training labels.

- [ ] **Step 3: Implement persistence helpers**

Create helpers to write `config.json`, append `metrics.jsonl`, write `events.json`, `summary.json`, and `fit_data.json`, and read a saved run.

- [ ] **Step 4: Run focused tests**

Run `pytest tests/test_training_dashboard.py -q`.

## Task 2: Streaming Training Runner

**Files:**
- Create/Modify: `src/training/runner.py`
- Modify: `tests/test_training_dashboard.py`

- [ ] **Step 1: Write tests for a short fixed run and EcoGrow event mapping**

Use small epoch counts to verify the runner returns summary, metrics, fit data, and valid event payloads.

- [ ] **Step 2: Implement run config defaults and train_strategy**

Implement one entry point that accepts a config dict, strategy name, run directory, and callback. It should train one strategy, emit one metric per epoch, persist artifacts, and return summary.

- [ ] **Step 3: Support Fixed, AdaGrid, and EcoGrow**

Fixed uses `DynamicKANLayer` with `fixed_grid`. AdaGrid uses `ExtendGridOnPlateau`. EcoGrow uses `EcoGrowScheduler` and maps scheduler events to frontend payloads.

- [ ] **Step 4: Run focused tests**

Run `pytest tests/test_training_dashboard.py -q`.

## Task 3: FastAPI Backend

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/schemas.py`
- Create: `backend/run_manager.py`
- Create: `backend/app.py`
- Modify: `requirements.txt`
- Test: `tests/test_backend_api.py`

- [ ] **Step 1: Add backend dependency tests**

Test `/api/health`, `/api/runs`, and `POST /api/runs` with a monkeypatched short run.

- [ ] **Step 2: Implement schemas and validation**

Normalize request config, clamp invalid numeric values with clear HTTP 400 errors, and reject unknown strategies.

- [ ] **Step 3: Implement RunManager**

Manage one active background thread, event queue, run IDs, saved run listing, full run loading, and SSE event formatting.

- [ ] **Step 4: Implement FastAPI app**

Serve `/`, `/api/health`, `/api/runs`, `/api/runs/{run_id}`, and `/api/runs/{run_id}/stream`.

- [ ] **Step 5: Run backend tests**

Run `pytest tests/test_backend_api.py -q`.

## Task 4: Refactor Existing HTML Frontend

**Files:**
- Modify: `visualization/ecogrow_live_comparison.html`

- [ ] **Step 1: Replace mojibake and simulation UI copy with clean Chinese**

Keep the dashboard concept but make copy readable and oriented around real backend training.

- [ ] **Step 2: Replace JS simulation state with API state**

Remove/stop using `KAN1D`, `AdaScheduler`, `EcoScheduler`, and browser-side training loops. Add `apiGet`, `apiPost`, `startRun`, `connectStream`, `loadRuns`, and `loadRunDetail`.

- [ ] **Step 3: Implement selected-run chart rendering**

Render train loss, validation loss, grid history, and final fit from selected run data.

- [ ] **Step 4: Implement health, controls, history, and event log**

Disable start when backend is unhealthy or a run is active. Add checkboxes for saved runs. Show current run metrics and events.

## Task 5: Integration Verification

**Files:**
- Modify as needed based on verification results.

- [ ] **Step 1: Run all tests**

Run `pytest -q`.

- [ ] **Step 2: Start backend**

Run `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`.

- [ ] **Step 3: Verify API manually**

Check `/api/health` and a short `POST /api/runs` request.

- [ ] **Step 4: Verify frontend manually**

Open `http://127.0.0.1:8000/`, start a short run, confirm live metrics, saved history, and checkbox chart toggling.
