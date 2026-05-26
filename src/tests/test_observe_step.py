"""
test_observe_step.py
====================
Unit tests for observe_step.py.

Run:
    python -m unittest discover src/tests
    # or
    pytest src/tests/
"""

import sys
import os
import time
import unittest

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observe_step import observe_step, PipelineObserver, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_observer() -> PipelineObserver:
    return PipelineObserver(pipeline_name="test_pipeline")


# ---------------------------------------------------------------------------
# @observe_step — bare form (no parentheses, no observer)
# ---------------------------------------------------------------------------

class TestObserveStepBareForm(unittest.TestCase):

    def test_return_value_is_passed_through(self):
        @observe_step
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)

    def test_exception_is_reraised(self):
        @observe_step
        def boom():
            raise RuntimeError("kaboom")

        with self.assertRaises(RuntimeError):
            boom()

    def test_function_name_is_preserved(self):
        @observe_step
        def my_step():
            pass

        self.assertEqual(my_step.__name__, "my_step")


# ---------------------------------------------------------------------------
# @observe_step(observer=...) — parameterised form
# ---------------------------------------------------------------------------

class TestObserveStepWithObserver(unittest.TestCase):

    def test_success_recorded_with_correct_fields(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def load_data():
            return {"rows": 100}

        load_data()

        self.assertEqual(len(observer.results), 1)
        r = observer.results[0]
        self.assertEqual(r.step_name, "load_data")
        self.assertEqual(r.status, "success")
        self.assertIsNone(r.error)
        self.assertIsNone(r.error_type)
        self.assertTrue(r.passed)

    def test_failure_recorded_with_error_info(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def bad_step():
            raise ValueError("shape mismatch")

        with self.assertRaises(ValueError):
            bad_step()

        self.assertEqual(len(observer.results), 1)
        r = observer.results[0]
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.error_type, "ValueError")
        self.assertIn("shape mismatch", r.error)
        self.assertFalse(r.passed)

    def test_exception_is_still_reraised_with_observer(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def explode():
            raise KeyError("missing_key")

        with self.assertRaises(KeyError):
            explode()

    def test_duration_ms_is_positive(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def slow_step():
            time.sleep(0.01)  # 10ms

        slow_step()

        r = observer.results[0]
        self.assertGreater(r.duration_ms, 0)

    def test_custom_name_overrides_function_name(self):
        observer = make_observer()

        @observe_step(observer=observer, name="custom_load")
        def load_features():
            return {}

        load_features()

        self.assertEqual(observer.results[0].step_name, "custom_load")

    def test_multiple_steps_accumulated_in_order(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def step_one():
            return 1

        @observe_step(observer=observer)
        def step_two():
            return 2

        step_one()
        step_two()

        self.assertEqual(len(observer.results), 2)
        self.assertEqual(observer.results[0].step_name, "step_one")
        self.assertEqual(observer.results[1].step_name, "step_two")

    def test_return_value_passed_through_with_observer(self):
        observer = make_observer()

        @observe_step(observer=observer)
        def compute():
            return 42

        result = compute()
        self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# PipelineObserver
# ---------------------------------------------------------------------------

class TestPipelineObserver(unittest.TestCase):

    def _make_result(self, name: str, status: str = "success", error: str = None) -> StepResult:
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        return StepResult(
            step_name=name,
            started_at=now,
            finished_at=now,
            duration_ms=10.0,
            status=status,
            error=error,
            error_type="ValueError" if error else None,
        )

    def test_all_passed_true_when_all_succeed(self):
        observer = make_observer()
        observer.record(self._make_result("a"))
        observer.record(self._make_result("b"))
        self.assertTrue(observer.all_passed())

    def test_all_passed_false_when_any_fail(self):
        observer = make_observer()
        observer.record(self._make_result("a"))
        observer.record(self._make_result("b", status="failed", error="oops"))
        self.assertFalse(observer.all_passed())

    def test_all_passed_false_on_empty(self):
        observer = make_observer()
        self.assertFalse(observer.all_passed())

    def test_failed_steps_returns_only_failures(self):
        observer = make_observer()
        observer.record(self._make_result("a"))
        observer.record(self._make_result("b", status="failed", error="bad"))
        observer.record(self._make_result("c"))

        failed = observer.failed_steps()
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].step_name, "b")

    def test_failed_steps_empty_when_all_pass(self):
        observer = make_observer()
        observer.record(self._make_result("a"))
        self.assertEqual(observer.failed_steps(), [])

    def test_print_summary_does_not_raise_on_empty(self):
        observer = make_observer()
        try:
            observer.print_summary()
        except Exception as exc:
            self.fail(f"print_summary() raised on empty observer: {exc}")

    def test_print_summary_does_not_raise_on_mixed(self):
        observer = make_observer()
        observer.record(self._make_result("a"))
        observer.record(self._make_result("b", status="failed", error="crash"))
        try:
            observer.print_summary()
        except Exception as exc:
            self.fail(f"print_summary() raised on mixed results: {exc}")


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult(unittest.TestCase):

    def _make(self, status: str = "success") -> StepResult:
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        return StepResult(
            step_name="my_step",
            started_at=now,
            finished_at=now,
            duration_ms=5.0,
            status=status,
        )

    def test_passed_true_on_success(self):
        self.assertTrue(self._make("success").passed)

    def test_passed_false_on_failed(self):
        self.assertFalse(self._make("failed").passed)

    def test_str_contains_pass_tag(self):
        self.assertIn("PASS", str(self._make("success")))

    def test_str_contains_fail_tag(self):
        self.assertIn("FAIL", str(self._make("failed")))


if __name__ == "__main__":
    unittest.main()
