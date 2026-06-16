"""
pipeline_example.py  [STARTER]
===============================
A churn prediction pipeline — no observability yet.

This is the "before" state. Run it and notice:
  - You get a final score but no step-level visibility
  - When something fails, you won't know which step, how long it ran, or why

Your job during the demo:
  1. Uncomment the observe_step import below
  2. Create a PipelineObserver instance
  3. Apply @observe_step(observer=observer) to each step function
  4. Call observer.print_summary() at the end of run_pipeline()
"""

import sys
import logging
from datetime import datetime, timedelta, timezone

sys.path.insert(
    0, __file__.rsplit("/starter", 1)[0] + "/starter"
)  # find freshness_validator and observe_step

from freshness_validator import DataSource, FreshnessValidator

# TODO 1 — uncomment when observe_step is ready:
from observe_step import observe_step, PipelineObserver

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s  %(name)s: %(message)s"
)

# ---------------------------------------------------------------------------
# TODO 2 — Create a PipelineObserver here, pass it to each @observe_step
# ---------------------------------------------------------------------------
observer = PipelineObserver(pipeline_name="churn_pipeline")


# ---------------------------------------------------------------------------
# Step functions — logic is complete, observability is missing
# ---------------------------------------------------------------------------


# TODO 3 — add @observe_step(observer=observer) here
@observe_step(observer=observer)
def load_features(customer_id: str, inject_failure: bool = False) -> dict:
    if inject_failure:
        raise ValueError("unexpected null in 'tenure_days'")
    return {
        "customer_id": customer_id,
        "tenure_days": 412,
        "monthly_spend": 89.50,
        "support_tickets_30d": 2,
        "last_login_days_ago": 3,
    }


# TODO 3 — add @observe_step(observer=observer) here
@observe_step(observer=observer)
def engineer_features(raw: dict) -> dict:
    return {
        "tenure_months": raw["tenure_days"] // 30,
        "high_support_usage": raw["support_tickets_30d"] >= 3,
        "engagement_score": max(0.0, 1.0 - raw["last_login_days_ago"] / 30),
        "monthly_spend": raw["monthly_spend"],
    }


# TODO 3 — add @observe_step(observer=observer) here
@observe_step(observer=observer)
def run_inference(features: dict) -> float:
    score = (
        0.3 * (1 - features["tenure_months"] / 24)
        + 0.4 * int(features["high_support_usage"])
        + 0.2 * (1 - features["engagement_score"])
        + 0.1 * (features["monthly_spend"] / 200)
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Data sources for freshness gate (Video 3 — already complete)
# ---------------------------------------------------------------------------


def make_sources(stale: bool = False) -> list:
    offset = timedelta(hours=3) if stale else timedelta(minutes=5)
    fresh_time = datetime.now(tz=timezone.utc) - offset
    return [
        DataSource(
            name="feature_store",
            expected_freshness=timedelta(hours=1),
            get_last_updated=lambda: fresh_time,
        ),
        DataSource(
            name="model_registry",
            expected_freshness=timedelta(hours=6),
            get_last_updated=lambda: (
                datetime.now(tz=timezone.utc) - timedelta(minutes=30)
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    customer_id: str, inject_failure: bool = False, stale_data: bool = False
):
    print(f"\n{'=' * 50}")
    print(f"Running pipeline for customer: {customer_id}")
    print(f"{'=' * 50}")

    # Freshness gate (Video 3 — already wired up)
    validator = FreshnessValidator()
    sources = make_sources(stale=stale_data)
    print("\n[Freshness Gate]")
    if not validator.all_fresh(sources):
        print("  BLOCKED — stale data detected. Aborting.\n")
        return None
    print("  All sources fresh — proceeding.\n")

    # Steps — logic works, but we're flying blind
    raw = None
    features = None
    score = None

    try:
        raw = load_features(customer_id, inject_failure=inject_failure)
    except Exception:
        pass

    if raw is not None:
        try:
            features = engineer_features(raw)
        except Exception:
            pass

    if features is not None:
        try:
            score = run_inference(features)
        except Exception:
            pass

    # TODO 4 — call observer.print_summary() here
    observer.print_summary()

    if score is not None:
        print(f"Churn probability for {customer_id}: {score:.2%}")
    else:
        print("Pipeline did not complete — check logs.")

    return score


# ---------------------------------------------------------------------------
# Run it — notice: result shows, but which steps ran? How long did they take?
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario 1 — happy path (no visibility)
    # run_pipeline("cust_A001")

    # TODO 5 — after adding observability, uncomment these:
    observer.results.clear()
    run_pipeline("cust_B999", inject_failure=True)  # see a FAIL row
