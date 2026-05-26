"""
pipeline_example.py
===================
End-to-end ML inference pipeline demo — combining FreshnessValidator (Video 3)
with step-level observability (Video 4).

What this shows
---------------
A 4-step churn prediction pipeline where:
  1. A freshness gate blocks stale data before any compute runs  (Video 3)
  2. Every downstream step is wrapped with @observe_step           (Video 4)
  3. A failure is intentionally injected so you can see FAIL rows in the summary
  4. observer.print_summary() gives a full picture of what ran, how long, and what broke

Run it:
    python src/examples/pipeline_example.py

Expected output (happy path):
    ── churn_pipeline ──────────────────────────────
    Step                      Status   Duration
    load_features             PASS     ...ms
    engineer_features         PASS     ...ms
    run_inference             PASS     ...ms
    ─────────────────────────────────────────────────
    3 steps | 3 passed | 0 failed

Expected output (injected failure in engineer_features):
    ── churn_pipeline ──────────────────────────────
    Step                      Status   Duration
    load_features             PASS     ...ms
    engineer_features         FAIL     ...ms   ValueError: unexpected null in 'tenure_days'
    run_inference             PASS     ...ms   (skipped because pipeline caught the error)
    ─────────────────────────────────────────────────
    3 steps | 2 passed | 1 failed
"""

import sys
import logging
from datetime import datetime, timedelta, timezone

# Add the src directory to the path so we can import without installing
sys.path.insert(0, __file__.rsplit("/examples", 1)[0])  # …/src
sys.path.insert(0, __file__.rsplit("/src/", 1)[0] + "/src")  # fallback

from freshness_validator import DataSource, FreshnessValidator
from observe_step import observe_step, PipelineObserver

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

observer = PipelineObserver(pipeline_name="churn_pipeline")


@observe_step(observer=observer)
def load_features(customer_id: str, inject_failure: bool = False) -> dict:
    """
    Pull raw features for a customer from the feature store.
    In production this hits a database; here we return a fake payload.
    """
    if inject_failure:
        raise ValueError("unexpected null in 'tenure_days'")

    return {
        "customer_id": customer_id,
        "tenure_days": 412,
        "monthly_spend": 89.50,
        "support_tickets_30d": 2,
        "last_login_days_ago": 3,
    }


@observe_step(observer=observer)
def engineer_features(raw: dict) -> dict:
    """
    Derive model-ready features from raw values.
    """
    return {
        "tenure_months": raw["tenure_days"] // 30,
        "high_support_usage": raw["support_tickets_30d"] >= 3,
        "engagement_score": max(0.0, 1.0 - raw["last_login_days_ago"] / 30),
        "monthly_spend": raw["monthly_spend"],
    }


@observe_step(observer=observer)
def run_inference(features: dict) -> float:
    """
    Run the churn model. Returns a probability in [0, 1].
    Fake linear combination — replace with your model.predict() call.
    """
    score = (
        0.3 * (1 - features["tenure_months"] / 24)
        + 0.4 * int(features["high_support_usage"])
        + 0.2 * (1 - features["engagement_score"])
        + 0.1 * (features["monthly_spend"] / 200)
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Data sources (for FreshnessValidator gate)
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
            get_last_updated=lambda: datetime.now(tz=timezone.utc) - timedelta(minutes=30),
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(customer_id: str, inject_failure: bool = False, stale_data: bool = False):
    print(f"\n{'='*50}")
    print(f"Running pipeline for customer: {customer_id}")
    if inject_failure:
        print("  [!] Failure injection enabled — load_features will raise")
    if stale_data:
        print("  [!] Stale data scenario — freshness gate will block")
    print(f"{'='*50}")

    # --- Step 0: Freshness gate (Video 3) ---
    validator = FreshnessValidator()
    sources = make_sources(stale=stale_data)

    print("\n[Freshness Gate]")
    if not validator.all_fresh(sources):
        print("  BLOCKED — stale data detected. Aborting pipeline.\n")
        return None

    print("  All sources fresh — proceeding.\n")

    # --- Step 1–3: Observed steps (Video 4) ---
    raw = None
    features = None
    score = None

    try:
        raw = load_features(customer_id, inject_failure=inject_failure)
    except Exception:
        pass  # StepResult already recorded; pipeline continues to summary

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

    # --- Summary ---
    observer.print_summary()

    if score is not None:
        print(f"Churn probability for {customer_id}: {score:.2%}")

    return score


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario 1 — happy path
    run_pipeline("cust_A001")

    # Reset observer between runs so we get clean summaries
    observer.results.clear()

    # Scenario 2 — step failure (feature load raises)
    run_pipeline("cust_B999", inject_failure=True)

    observer.results.clear()

    # Scenario 3 — freshness gate blocks (stale data)
    run_pipeline("cust_C123", stale_data=True)
