# Step-Level Observability for ML Pipelines

> Companion code for: **You Can't Debug What You Can't See — Adding Step-Level Observability to Your ML Pipeline**
> Part of the [Debugging ML in Production](https://www.youtube.com/playlist?list=PLOszX3Fd4bgccKUnq6cBaZbQ3V7245isU) series on YouTube.

---

## What this builds

A lightweight `@observe_step` decorator and `PipelineObserver` that wraps every step of your ML pipeline — capturing timing, status, and failure details with no external dependencies.

This is the hands-on fix for **Failure Mode 3: Observability Blindness**, introduced in [Part 1 of Debugging ML in Production](https://youtu.be/GsxQQvXGzDs).

```python
from src.observe_step import observe_step, PipelineObserver

observer = PipelineObserver(pipeline_name="churn_pipeline")

@observe_step(observer=observer)
def load_features(customer_id: str) -> dict:
    ...

@observe_step(observer=observer)
def run_inference(features: dict) -> float:
    ...

load_features("cust_123")
run_inference(features)

observer.print_summary()
# ── churn_pipeline ──────────────────────────────
# Step                      Status   Duration
# load_features             PASS     42.3ms
# run_inference             PASS     11.1ms
# ─────────────────────────────────────────────────
# 2 steps | 2 passed | 0 failed
```

---

## What you'll learn

- Why "the pipeline failed" tells you nothing useful — and how to fix that
- How to build a decorator that captures timing, status, and exceptions per step
- How to accumulate results across a full run into a structured summary
- How `@observe_step` and `FreshnessValidator` (Video 3) work together as a production observability stack

---

## Getting started

**Requirements:** Python 3.11+, no external dependencies for the core module.

```bash
git clone https://github.com/ai-engineering-with-peeush/yt-04-step-level-observability.git
cd yt-04-step-level-observability

# Run the end-to-end pipeline demo (3 scenarios: happy path, failure, stale data)
python src/examples/pipeline_example.py

# Run the test suite
python -m unittest discover src/tests
```

---

## Code structure

```
src/
  ├── observe_step.py          # Core: @observe_step decorator, StepResult, PipelineObserver
  ├── freshness_validator.py   # From Video 3 — data freshness gate (used in pipeline demo)
  ├── requirements.txt
  ├── examples/
  │   └── pipeline_example.py  # End-to-end demo — 3 scenarios (happy, failure, stale)
  └── tests/
      └── test_observe_step.py # 20 unit tests
```

### Key classes

| Class | Description |
|-------|-------------|
| `StepResult` | Immutable record of one step's outcome — timing, status, error info |
| `PipelineObserver` | Accumulates StepResults; prints a structured summary table |
| `observe_step` | Decorator — bare or parameterised; captures timing and exceptions |

---

## Series context

This repo is part of the **Debugging ML in Production** series:

| Video | Link |
|-------|------|
| Part 1 — 5 Failure Modes (Theory) | [Watch](https://youtu.be/GsxQQvXGzDs) |
| Part 2 — Failure Modes 4 & 5 (Theory) | [Watch](https://youtu.be/j_IjpiZE_4k) |
| Video 3 — Data Freshness Validator | [Watch](https://youtu.be/xW1xgx28D_w) |
| **Video 4 — Step-Level Observability (this repo)** | Coming June 2026 |

---

## Channel

**[AI Engineering with Peeush](https://www.youtube.com/@AIEngineeringWithPeeush)** — hands-on production ML engineering, one video at a time.
