"""
observe_step.py  [STARTER]
==========================
Step-level observability for ML pipelines.

Build this live during the demo. The data models are pre-defined below
(boilerplate) — your job is to implement the decorator and observer methods.

Demo build order:
  1. _wrap()               ← timing + try/except + StepResult creation
  2. observe_step()        ← dual-form decorator (bare vs parameterised)
  3. PipelineObserver:
       all_passed()
       failed_steps()
       print_summary()
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models — pre-defined, don't need to type these live
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step_name: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    status: str  # "success" | "failed"
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


class PipelineObserver:
    def __init__(self, pipeline_name: str = "pipeline"):
        self.pipeline_name = pipeline_name
        self.results: List[StepResult] = []

    def record(self, result: StepResult) -> None:
        self.results.append(result)
        if result.passed:
            logger.info(str(result))
        else:
            logger.error(str(result))

    # -------------------------------------------------------------------------
    # TODO 3a — Return True only if every recorded step succeeded.
    #            Edge case: return False when results list is empty.
    # -------------------------------------------------------------------------
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    # -------------------------------------------------------------------------
    # TODO 3b — Return all StepResults whose status is "failed".
    # -------------------------------------------------------------------------
    def failed_steps(self) -> List[StepResult]:
        return [r for r in self.results if not r.passed]

    # -------------------------------------------------------------------------
    # TODO 3c — Print a table of all observed steps.
    #
    # Target output:
    #   ── churn_pipeline ───────────────────────────────
    #   Step                      Status   Duration
    #   ─────────────────────────────────────────────────
    #   load_features             PASS     42.3ms
    #   engineer_features         FAIL     8.1ms   ValueError: shape mismatch
    #   ─────────────────────────────────────────────────
    #   2 steps | 1 passed | 1 failed
    # -------------------------------------------------------------------------
    def print_summary(self) -> None:
        width = 49
        print(
            f"\n── {self.pipeline_name} {'─' * (width - len(self.pipeline_name) - 4)}"
        )
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
        print(
            f"{total} step{'s' if total != 1 else ''} | {passed} passed | {failed} failed\n"
        )


# ---------------------------------------------------------------------------
# TODO 1 — _wrap(fn, observer, name)
#
# This is the actual implementation that runs around every pipeline step.
# It should:
#   - Record started_at before calling fn()
#   - Call fn(*args, **kwargs) inside a try/except
#   - Record finished_at and calculate duration_ms
#   - Build a StepResult (status="success" or "failed")
#   - If an observer is provided, call observer.record(result)
#     otherwise just log the result directly
#   - Re-raise any exception — don't swallow it
#   - Return the function's return value on success
#
# Use @functools.wraps(fn) to preserve the original function's metadata.
# ---------------------------------------------------------------------------
def _wrap(
    fn: Callable, observer: Optional[PipelineObserver], name: Optional[str]
) -> Callable:
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
            log = logger.info if step_result.passed else logger.error
            log(str(step_result))

        if exc_to_raise is not None:
            raise exc_to_raise

        return result_value

    return wrapper


# ---------------------------------------------------------------------------
# TODO 2 — observe_step(fn, *, observer, name)
#
# Needs to support TWO calling forms:
#
#   Bare (no parens):
#     @observe_step
#     def my_step(): ...
#
#   Parameterised:
#     @observe_step(observer=observer, name="custom")
#     def my_step(): ...
#
# Hint: when fn is None, the decorator was called with parens — return a
# decorator function. When fn is provided, wrap it directly.
# ---------------------------------------------------------------------------
def observe_step(
    fn: Optional[Callable] = None,
    *,
    observer: Optional[PipelineObserver] = None,
    name: Optional[str] = None,
):
    if fn is None:

        def decorator(f: Callable) -> Callable:
            return _wrap(f, observer=observer, name=name)

        return decorator
    return _wrap(fn, observer=observer, name=name)
