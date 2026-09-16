"""
Anomaly detection over the ticket dataset.

Design choice: anomaly detection is deterministic (statistics + business
rules), NOT LLM-based. Reasons:
  - Reproducibility: the same data always yields the same flags, which
    matters for something evaluators will check by hand.
  - Explainability: every flag carries a concrete numeric reason
    ("resolution time 87.2h vs threshold 53.0h"), which an LLM free-text
    judgement wouldn't reliably give without extra scaffolding.
  - Cost/latency: no LLM call needed to scan 500 rows.
The LLM is reserved for natural-language understanding (Section 2's other
requirement), where it's actually necessary.

Two anomaly types are detected:
  1. long_resolution  - resolution_time_hrs is a statistical outlier
                         among resolved tickets (IQR method: above
                         Q3 + 1.5 * IQR).
  2. sla_breach        - unresolved (Open/Escalated) ticket with
                         priority High or Critical, older than a
                         configurable age threshold (default 24h),
                         per the assessment's own example.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from app.db import get_dataframe

DEFAULT_SLA_HOURS = 24
DEFAULT_SLA_PRIORITIES = ("High", "Critical")


def _iqr_outlier_threshold(series: pd.Series) -> tuple[float, float, float]:
    """Return (q1, q3, upper_threshold) using the standard 1.5*IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return q1, q3, upper


def detect_long_resolution_outliers(df: pd.DataFrame) -> list[dict]:
    resolved = df[df["status"] == "Resolved"].dropna(subset=["resolution_time_hrs"])
    if resolved.empty:
        return []

    q1, q3, upper = _iqr_outlier_threshold(resolved["resolution_time_hrs"])
    flagged = resolved[resolved["resolution_time_hrs"] > upper]

    results = []
    for _, row in flagged.iterrows():
        results.append({
            "ticket_id": row["ticket_id"],
            "anomaly_type": "long_resolution",
            "priority": row["priority"],
            "category": row["category"],
            "agent_id": row["agent_id"],
            "resolution_time_hrs": row["resolution_time_hrs"],
            "reason": (
                f"Resolution time {row['resolution_time_hrs']:.1f}h exceeds "
                f"the normal range (upper threshold {upper:.1f}h, based on "
                f"IQR of resolved tickets: Q1={q1:.1f}h, Q3={q3:.1f}h)."
            ),
        })
    return results


def detect_sla_breaches(
    df: pd.DataFrame,
    as_of: Optional[datetime] = None,
    sla_hours: float = DEFAULT_SLA_HOURS,
    priorities: tuple = DEFAULT_SLA_PRIORITIES,
) -> list[dict]:
    """
    Flag unresolved High/Critical tickets older than `sla_hours`.

    `as_of` is the reference "current time". It defaults to the latest
    created_at timestamp in the dataset, since the dataset is historical
    (Jan-Mar 2024) and using the real wall-clock "now" would flag almost
    every ticket as a breach. An evaluator can override this via the API
    to test against a specific point in time.
    """
    if as_of is None:
        as_of = pd.to_datetime(df["created_at"]).max()
    else:
        as_of = pd.to_datetime(as_of)

    unresolved = df[df["status"].isin(["Open", "Escalated"])].copy()
    unresolved = unresolved[unresolved["priority"].isin(priorities)]
    unresolved["created_at_dt"] = pd.to_datetime(unresolved["created_at"])
    unresolved["age_hrs"] = (as_of - unresolved["created_at_dt"]).dt.total_seconds() / 3600

    breached = unresolved[unresolved["age_hrs"] > sla_hours]

    results = []
    for _, row in breached.iterrows():
        results.append({
            "ticket_id": row["ticket_id"],
            "anomaly_type": "sla_breach",
            "priority": row["priority"],
            "category": row["category"],
            "status": row["status"],
            "agent_id": row["agent_id"],
            "age_hrs": round(row["age_hrs"], 1),
            "reason": (
                f"{row['priority']} priority ticket has been {row['status'].lower()} "
                f"for {row['age_hrs']:.1f}h, exceeding the {sla_hours}h SLA threshold."
            ),
        })
    return results


def detect_anomalies(as_of: Optional[str] = None, sla_hours: float = DEFAULT_SLA_HOURS) -> dict:
    df = get_dataframe()
    as_of_dt = pd.to_datetime(as_of) if as_of else None

    long_res = detect_long_resolution_outliers(df)
    sla = detect_sla_breaches(df, as_of=as_of_dt, sla_hours=sla_hours)

    return {
        "as_of": str(as_of_dt) if as_of_dt is not None else str(pd.to_datetime(df["created_at"]).max()),
        "sla_hours_threshold": sla_hours,
        "total_flagged": len(long_res) + len(sla),
        "long_resolution_outliers": long_res,
        "sla_breaches": sla,
    }
