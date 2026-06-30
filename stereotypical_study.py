"""Analysis helpers for the stereotypical bias study.

SR_p = j's share of the actual positive class (TP+FN): (j_tp+j_fn) / total actual positives.
SR_n = j's share of the actual negative class (TN+FP): (j_tn+j_fp) / total actual negatives.
SR  = √(SR_p × SR_n) — geometric mean (combined stereotypical ratio).
SR = GR_j is the proportional reference for all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_analysis import ensure_metric_columns

SR_COLUMNS = {
    "sr_p": "stereotypical_ratio",
    "sr_n": "stereotypical_ratio_negative",
    "sr_c": "stereotypical_ratio_combined",
}

SR_LABELS = {
    "stereotypical_ratio": "SR_p (positive)",
    "stereotypical_ratio_negative": "SR_n (negative)",
    "stereotypical_ratio_combined": "SR_c (combined)",
}


def _ensure_sr_cols(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Ensure SR columns are present, computing via the registry if missing."""
    return ensure_metric_columns(df, list(cols))


def _filter_by_fixed(
    df: pd.DataFrame,
    ir_value: float | None,
    gr_value: float | None,
    atol: float,
) -> pd.DataFrame:
    if ir_value is not None:
        mask = np.isclose(df["imbalance_ratio"].to_numpy(np.float64), ir_value, atol=atol, rtol=0)
        df = df[mask].reset_index(drop=True)
    if gr_value is not None:
        mask = np.isclose(df["group_ratio_j"].to_numpy(np.float64), gr_value, atol=atol, rtol=0)
        df = df[mask].reset_index(drop=True)
    return df


def metric_means_by_sr(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    ir_value: float | None = None,
    gr_value: float | None = None,
    atol: float = 0.01,
    absolute: bool = False,
) -> pd.DataFrame:
    """Per-SR-value mean and std for each metric."""
    work = _ensure_sr_cols(df, sr_col)
    work = _filter_by_fixed(work, ir_value, gr_value, atol)
    work = ensure_metric_columns(work, metric_keys)

    rows: list[dict] = []
    for sr_val, group in work.groupby(sr_col, sort=True):
        row: dict = {"sr": float(sr_val)}
        for key in metric_keys:
            vals = group[key].abs() if absolute else group[key]
            valid = vals.dropna()
            row[key] = float(valid.mean()) if len(valid) > 0 else np.nan
            row[f"{key}_std"] = float(valid.std()) if len(valid) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def metric_means_by_sr_multi_ir(
    df: pd.DataFrame,
    metric_key: str,
    ir_values: list[float],
    *,
    sr_col: str = "stereotypical_ratio",
    gr_value: float | None = None,
    atol: float = 0.01,
    absolute: bool = False,
) -> pd.DataFrame:
    """Per-(SR, IR) mean and std for a single metric at a fixed GR.

    Columns: ``sr``, ``ir_value``, ``{metric_key}``, ``{metric_key}_std``.
    """
    rows: list[dict] = []
    for ir_val in ir_values:
        slice_df = metric_means_by_sr(
            df, [metric_key],
            sr_col=sr_col, ir_value=ir_val, gr_value=gr_value, atol=atol, absolute=absolute,
        )
        for _, row in slice_df.iterrows():
            rows.append({
                "sr": row["sr"],
                "ir_value": float(ir_val),
                metric_key: row.get(metric_key, np.nan),
                f"{metric_key}_std": row.get(f"{metric_key}_std", np.nan),
            })
    return pd.DataFrame(rows)


def proportional_sr_slice(
    df: pd.DataFrame,
    metric_keys: list[str],
    *,
    sr_col: str = "stereotypical_ratio",
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Rows where SR ≈ GR_j (proportional prediction) within *tolerance*."""
    work = _ensure_sr_cols(df, sr_col)
    sr = work[sr_col].to_numpy(np.float64)
    gr_j = work["group_ratio_j"].to_numpy(np.float64)
    mask = np.abs(sr - gr_j) <= tolerance
    return ensure_metric_columns(work[mask].reset_index(drop=True), metric_keys)
