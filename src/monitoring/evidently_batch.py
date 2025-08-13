# src/monitoring/evidently_batch.py
import argparse
import logging
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Evidently 0.4+ API
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

LOG = logging.getLogger("evidently_batch")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Defaults (can be overridden by CLI/env)
DEFAULT_CURRENT = "data/processed/clean_data.csv"
DEFAULT_REFERENCE = "data/reference/clean_data_ref.csv"
DEFAULT_PUSHGATEWAY = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
DEFAULT_INSTANCE = os.getenv("INSTANCE", "local")


def _extract_metrics_as_numbers(report_dict: Dict[str, Any]) -> Dict[str, float]:
    """Pull a few stable numeric fields from Evidently (0.4+) report dict."""
    out: Dict[str, float] = {}
    metrics_list = report_dict.get("metrics", [])

    # Drift bits
    drift_share = None
    n_drifted = None
    data_drift = None

    # Summary bits (row count, missing)
    current_rows = None
    missing_total = None

    for m in metrics_list:
        metric_name = m.get("metric")
        res = m.get("result", {}) or {}

        # Data drift
        if metric_name in (
            "DataDriftPreset",
            "DataDriftMetric",
            "DatasetDriftMetric",
            "DataDriftTable",
        ):
            drift_share = drift_share or res.get("share_of_drifted_columns")
            n_drifted = n_drifted or res.get("number_of_drifted_columns")
            data_drift = data_drift or res.get("dataset_drift") or res.get("drift_detected")

        # Data summary (current stats live under result.current)
        if metric_name in (
            "DataSummaryPreset",
            "DataSummaryMetric",
            "DataSummaryTable",
        ):
            cur = res.get("current", {}) or {}
            current_rows = current_rows or cur.get("number_of_rows") or cur.get("n_rows")
            missing_total = (
                missing_total or cur.get("number_of_missing_values") or cur.get("n_missing_values")
            )

    out["evidently_drift_share"] = float(drift_share) if drift_share is not None else 0.0
    out["evidently_features_drifted_total"] = float(n_drifted) if n_drifted is not None else 0.0
    out["evidently_data_drift"] = 1.0 if bool(data_drift) else 0.0
    out["evidently_current_rows_total"] = float(current_rows) if current_rows is not None else 0.0
    out["evidently_missing_values_total"] = (
        float(missing_total) if missing_total is not None else 0.0
    )
    return out


def _push_metrics(
    metrics: Dict[str, float], job: str, gateway: str, grouping: Dict[str, str]
) -> None:
    """Push gauges to Pushgateway; don't raise."""
    reg = CollectorRegistry()
    gauges = {name: Gauge(name, f"{name} (evidently)", registry=reg) for name in metrics}
    for k, v in metrics.items():
        gauges[k].set(float(v))
    push_to_gateway(gateway, job=job, grouping_key=grouping, registry=reg)


def run(
    current_csv: str = DEFAULT_CURRENT,
    reference_csv: str = DEFAULT_REFERENCE,
    pushgateway: str = DEFAULT_PUSHGATEWAY,
    instance: str = DEFAULT_INSTANCE,
    strict: bool = False,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    LOG.info(f"Loading reference: {reference_csv}")
    LOG.info(f"Loading current:   {current_csv}")

    if not os.path.exists(reference_csv):
        msg = f"Reference CSV not found: {reference_csv}"
        LOG.error(msg)
        if strict:
            raise FileNotFoundError(msg)
        return {}

    if not os.path.exists(current_csv):
        msg = f"Current CSV not found: {current_csv}"
        LOG.error(msg)
        if strict:
            raise FileNotFoundError(msg)
        return {}

    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)

    # Build & run report (drift + summary)
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    evaluation = report.run(current_data=cur, reference_data=ref)

    # 0.4+ -> evaluation.dict()
    try:
        as_dict = evaluation.dict()
    except AttributeError:
        # very defensive fallback
        as_dict = report.as_dict()

    numbers = _extract_metrics_as_numbers(as_dict)
    numbers.setdefault("evidently_run_duration_seconds", float(time.perf_counter() - t0))

    grouping = {"instance": instance, "service": "evidently"}
    try:
        _push_metrics(numbers, job="evidently_batch", gateway=pushgateway, grouping=grouping)
        LOG.info(f"Pushed to {pushgateway}: {numbers}")
    except Exception as e:
        LOG.warning(f"Failed to push to Pushgateway: {e}")
        if strict:
            raise

    return numbers


def main():
    p = argparse.ArgumentParser(description="Run Evidently drift/summary and push metrics.")
    p.add_argument(
        "--current",
        default=DEFAULT_CURRENT,
        help="Path to current CSV (default: data/processed/clean_data.csv)",
    )
    p.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Path to reference CSV (default: data/reference/clean_data_ref.csv)",
    )
    p.add_argument(
        "--pushgateway",
        default=DEFAULT_PUSHGATEWAY,
        help="Pushgateway URL (default: http://pushgateway:9091)",
    )
    p.add_argument(
        "--instance",
        default=DEFAULT_INSTANCE,
        help='Grouping label "instance" (default: local)',
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise on errors (push/IO); default soft-fails.",
    )
    args = p.parse_args()

    try:
        res = run(
            current_csv=args.current,
            reference_csv=args.reference,
            pushgateway=args.pushgateway,
            instance=args.instance,
            strict=args.strict,
        )
        LOG.info(f"Done. Metrics: {res}")
    except Exception as e:
        LOG.exception(f"Run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
