"""
observe_step.py
===============
Step-level observability for ML pipelines — no external dependencies.

Companion code for: "You Can't Debug What You Can't See — Adding Step-Level
Observability to Your ML Pipeline" (AI Engineering with Peeush, Video 4)

Problem it solves
-----------------
Your ML pipeline is a black box in production. When it fails, you know *that*
something went wrong, but not *which step*, how long it ran, or what it
returned before the crash. This module adds a lightweight @observe_step
decorator that captures per-step timing, status, and failure info — no config,
no infrastructure, no external libraries.

Usage
-----
    from observe_step import observe_step, PipelineObserver

    observer = PipelineObserver()

    @observe_step(observer=observer)
    def load_features(customer_id: str) -> dict:
        ...

    @observe_step(observer=observer)
    def run_inference(features: dict) -> float:
        ...

    load_features("cust_123")
    score = run_inference(features)

    observer.print_summary()
    # Step                  Status   Duration
    # load_features         PASS     42ms
    # run_inference         PASS     11ms

Bare decorator form (observer-less, just logs):
    @observe_step
    def preprocess(df):
        ...
"""

from __future__ import annotations

import functools
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """
    The outcome of a single observed pipeline step.

    Attributes
    ----------
    step_name : str
        Name of the step — defaults to the wrapped function's __name__.
    started_at : datetime
        UTC timestamp when the step began executing.
    finished_at : datetime
        UTC timestamp when the step completed (success or failure).
    duration_ms : float
        Wall-clock time for the step in milliseconds.
    status : str
        "success" | "failed"
    error : str | None
        The exception message if status == "failed", else None.
    error_type : str | None
        The exception class name (e.g. "ValueError") if status == "failed".
    """
    step_name: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    status: str          # "success" | "failed"
    error: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == "success"

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        base = f"[{tag}] {self.step_name} — {self.duration_ms:.1f}ms"
        if self.error:
            base += f" | {self.error_type}: {self.error}"
        return base


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class PipelineObserver:
    """
    Collects StepResults across an entire pipeline run.

    Pass a single PipelineObserver instance to every @observe_step decorator
    in your pipeline. After the run, call print_summary() to see a structured
    view of every step's status and duration.

    Example
    -------
        observer = PipelineObserver(pipeline_name="churn_inference")

        @observe_step(observer=observer)
        def load_features(...): ...

        @observe_step(observer=observer)
        def run_model(...): ...

        ...

        observer.print_summary()
    """

    def __init__(self, pipeline_name: str = "pipeline"):
        self.pipeline_name = pipeline_name
        self.results: List[StepResult] = []

    def record(self, result: StepResult) -> None:
        """Append a StepResult and emit a log line."""
        self.results.append(result)
        if result.passed:
            logger.info(str(result))
        else:
            logger.error(str(result))

    def all_passed(self) -> bool:
        """Return True only if every recorded step succeeded."""
        return bool(self.results) and all(r.passed for r in self.results)

    def failed_steps(self) -> List[StepResult]:
        """Return all StepResults whose status is 'failed'."""
        return [r for r in self.results if not r.passed]

    def print_summary(self) -> None:
        """
        Print a table of all observed steps to stdout.

        Example output:
            ── churn_inference ──────────────────────────────
            Step                   Status   Duration
            load_features          PASS     42.3ms
            run_model              FAIL     8.1ms   ValueError: shape mismatch
            ─────────────────────────────────────────────────
            2 steps | 1 passed | 1 failed
        """
        width = 49
        print(f"\n── {self.pipeline_name} {'─' * (width - len(self.pipeline_name) - 4)}")
        print(f"{'Step':<25} {'Status':<8} {'Duration'}")
        print("─" * width)

        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            row = f"{r.step_name:<25} {tag:<8} {r.duration_ms:.1f}ms"
            if r.error:
                row += f"   {r.error_type}: {r.error}"
            print(row)

        print("─" * width)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"{total} step{'s' if total != 1 else ''} | {passed} passed | {failed} failed\n")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def observe_step(
    fn: Optional[Callable] = None,
    *,
    observer: Optional[PipelineObserver] = None,
    name: Optional[str] = None,
):
    """
    Decorator that wraps a pipeline step with timing and status capture.

    Supports two calling forms:

    Bare (no parentheses) — logs each call, no observer accumulation:
        @observe_step
        def my_step(): ...

    Parameterised — attaches to a PipelineObserver:
        @observe_step(observer=observer, name="custom_name")
        def my_step(): ...

    Parameters
    ----------
    fn : Callable, optional
        The function being decorated (set automatically in bare form).
    observer : PipelineObserver, optional
        If provided, StepResult is appended to the observer's result list.
    name : str, optional
        Override the step name in logs/results. Defaults to fn.__name__.

    Behaviour
    ---------
    - The decorated function's return value is passed through unchanged.
    - If the function raises, the exception is recorded in the StepResult
      (status="failed") and then **re-raised** — the decorator does not swallow
      errors. Your pipeline's error handling stays in control.
    """
    if fn is None:
        # Called as @observe_step(observer=...) — return the actual decorator
        def decorator(f: Callable) -> Callable:
            return _wrap(f, observer=observer, name=name)
        return decorator

    # Called as @observe_step (no parentheses)
    return _wrap(fn, observer=observer, name=name)


def _wrap(fn: Callable, observer: Optional[PipelineObserver], name: Optional[str]) -> Callable:
    step_name = name or fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started_at = datetime.now(tz=timezone.utc)
        exc_to_raise = None
        result_value = None
        status = "success"
        error: Optional[str] = None
        error_type: Optional[str] = None

        try:
            result_value = fn(*args, **kwargs)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            error_type = type(exc).__name__
            exc_to_raise = exc

        finished_at = datetime.now(tz=timezone.utc)
        duration_ms = (finished_at - started_at).total_seconds() * 1000

        step_result = StepResult(
            step_name=step_name,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            status=status,
            error=error,
            error_type=error_type,
        )

        if observer is not None:
            observer.record(step_result)
        else:
            # No observer — just log
            tag = "PASS" if step_result.passed else "FAIL"
            log = logger.info if step_result.passed else logger.error
            log(str(step_result))

        if exc_to_raise is not None:
            raise exc_to_raise

        return result_value

    return wrapper
