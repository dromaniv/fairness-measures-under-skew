"""Built-in fairness, ratio, and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metric_registry import MetricSpec, register_metric, safe_divide


def _i_total(df: pd.DataFrame) -> pd.Series:
    return df["i_tp"] + df["i_fp"] + df["i_tn"] + df["i_fn"]


def _j_total(df: pd.DataFrame) -> pd.Series:
    return df["j_tp"] + df["j_fp"] + df["j_tn"] + df["j_fn"]


def _total(df: pd.DataFrame) -> pd.Series:
    return _i_total(df) + _j_total(df)


def _positive_total(df: pd.DataFrame) -> pd.Series:
    return df["i_tp"] + df["i_fn"] + df["j_tp"] + df["j_fn"]


def _negative_total(df: pd.DataFrame) -> pd.Series:
    return df["i_fp"] + df["i_tn"] + df["j_fp"] + df["j_tn"]


def group_ratio_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_j_total(df), _total(df))


def group_ratio_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_i_total(df), _total(df))


def imbalance_ratio(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(_positive_total(df), _total(df))


def accuracy(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"] + df["i_tn"] + df["j_tp"] + df["j_tn"], _total(df))


def g_mean(df: pd.DataFrame) -> np.ndarray:
    tpr = safe_divide(df["i_tp"] + df["j_tp"], _positive_total(df))
    tnr = safe_divide(df["i_tn"] + df["j_tn"], _negative_total(df))
    return np.sqrt(tpr * tnr)


def tpr_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"], df["i_tp"] + df["i_fn"])


def tpr_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"], df["j_tp"] + df["j_fn"])


def fpr_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_fp"], df["i_fp"] + df["i_tn"])


def fpr_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_fp"], df["j_fp"] + df["j_tn"])


def ppv_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tp"], df["i_tp"] + df["i_fp"])


def ppv_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"], df["j_tp"] + df["j_fp"])


def npv_i(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["i_tn"], df["i_tn"] + df["i_fn"])


def npv_j(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tn"], df["j_tn"] + df["j_fn"])


def acc_equality_diff(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"] + df["j_tn"], _j_total(df)) - safe_divide(df["i_tp"] + df["i_tn"], _i_total(df))


def statistical_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return safe_divide(df["j_tp"] + df["j_fp"], _j_total(df)) - safe_divide(df["i_tp"] + df["i_fp"], _i_total(df))


def equal_opportunity_diff(df: pd.DataFrame) -> np.ndarray:
    return tpr_j(df) - tpr_i(df)


def predictive_equality_diff(df: pd.DataFrame) -> np.ndarray:
    return fpr_j(df) - fpr_i(df)


def positive_predictive_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return ppv_j(df) - ppv_i(df)


def negative_predictive_parity_diff(df: pd.DataFrame) -> np.ndarray:
    return npv_j(df) - npv_i(df)


def equalized_odds_diff(df: pd.DataFrame) -> np.ndarray:
    """Equalized Odds Difference: max of |TPR gap| and |FPR gap| between groups."""
    tpr_gap = np.abs(np.asarray(tpr_j(df), dtype=np.float64) - np.asarray(tpr_i(df), dtype=np.float64))
    fpr_gap = np.abs(np.asarray(fpr_j(df), dtype=np.float64) - np.asarray(fpr_i(df), dtype=np.float64))
    result = np.full(len(df), np.nan, dtype=np.float64)
    both_defined = ~(np.isnan(tpr_gap) | np.isnan(fpr_gap))
    result[both_defined] = np.maximum(tpr_gap[both_defined], fpr_gap[both_defined])
    return result

def stereotypical_ratio(df: pd.DataFrame) -> np.ndarray:
    """SR_p = j's share of the actual positive class (TP+FN): (j_tp+j_fn) / total actual positives.

    SR_p = GR_j means proportional representation in the positive class.
    NaN when no actual positives exist.
    """
    total_ap = np.asarray(df["i_tp"] + df["i_fn"] + df["j_tp"] + df["j_fn"], dtype=np.float64)
    j_ap = np.asarray(df["j_tp"] + df["j_fn"], dtype=np.float64)
    return safe_divide(j_ap, total_ap)


def stereotypical_ratio_negative(df: pd.DataFrame) -> np.ndarray:
    """SR_n = j's share of the actual negative class (TN+FP): (j_tn+j_fp) / total actual negatives.

    SR_n = GR_j means proportional representation in the negative class.
    NaN when no actual negatives exist.
    """
    total_an = np.asarray(df["i_tn"] + df["i_fp"] + df["j_tn"] + df["j_fp"], dtype=np.float64)
    j_an = np.asarray(df["j_tn"] + df["j_fp"], dtype=np.float64)
    return safe_divide(j_an, total_an)


def stereotypical_ratio_combined(df: pd.DataFrame) -> np.ndarray:
    """SR = √(SR_p × SR_n): geometric mean of positive and negative class representation shares.

    SR = GR_j when both class representations are proportional.
    NaN when either SR_p or SR_n is undefined.
    """
    sr_p = np.asarray(stereotypical_ratio(df), dtype=np.float64)
    sr_n = np.asarray(stereotypical_ratio_negative(df), dtype=np.float64)
    out = np.full(len(df), np.nan, dtype=np.float64)
    valid = np.isfinite(sr_p) & np.isfinite(sr_n) & (sr_p >= 0) & (sr_n >= 0)
    out[valid] = np.sqrt(sr_p[valid] * sr_n[valid])
    return out

def register_builtin_metrics() -> None:
    specs = [
        MetricSpec(
            key="stereotypical_ratio",
            label="Stereotypical Ratio SR_p",
            category="ratio",
            sort_order=0,
            description=(
                "SR_p: j's share of the actual positive class (TP+FN). "
                "SR_p = GR_j means proportional representation in positives (no stereotypical bias). "
                "Primary SR axis: only PPV/NPV-based fairness metrics show substantial correlation with SR_p "
                "(stratified Spearman |ρ| ≈ 0.74); all other metrics are near-insensitive."
            ),
            formula=r"\mathrm{SR}_p = \frac{j_\mathrm{tp}+j_\mathrm{fn}}{i_\mathrm{tp}+i_\mathrm{fn}+j_\mathrm{tp}+j_\mathrm{fn}}",
            compute=stereotypical_ratio,
        ),
        MetricSpec(
            key="stereotypical_ratio_negative",
            label="Stereotypical Ratio SR_n",
            category="ratio",
            sort_order=1,
            description=(
                "SR_n: j's share of the actual negative class (TN+FP). "
                "SR_n = GR_j means proportional representation in negatives. "
                "Algebraically determined by SR_p and IR: SR_n − GR = −(SR_p − GR) · IR/(1−IR). "
                "SR_n adds no information beyond SR_p given IR; its fairness-metric correlations are "
                "the exact sign-flip of SR_p's."
            ),
            formula=r"\mathrm{SR}_n = \frac{j_\mathrm{tn}+j_\mathrm{fp}}{i_\mathrm{tn}+i_\mathrm{fp}+j_\mathrm{tn}+j_\mathrm{fp}}",
            compute=stereotypical_ratio_negative,
        ),
        MetricSpec(
            key="stereotypical_ratio_combined",
            label="Stereotypical Ratio SR_c",
            category="ratio",
            sort_order=2,
            description=(
                "SR_c = √(SR_p × SR_n): geometric mean of positive and negative class representation shares. "
                "SR_c = GR_j only at the proportional neutral point. "
                "Directionally blind: because SR_p and SR_n always deviate from GR_j in opposite directions, "
                "their geometric mean collapses the sign of stereotypical bias "
                "(at IR = 0.5, SR_c ≤ GR_j always regardless of direction). "
                "Stratified Spearman |ρ| ≈ 0 with all fairness metrics; use SR_p instead."
            ),
            formula=r"\mathrm{SR}_c = \sqrt{\mathrm{SR}_p \cdot \mathrm{SR}_n}",
            compute=stereotypical_ratio_combined,
        ),
        MetricSpec(
            key="group_ratio_j",
            label="Group Ratio (j / total)",
            category="ratio",
            description="Group ratio: j-group count divided by total count.",
            formula="(j_tp + j_fp + j_tn + j_fn) / total",
            compute=group_ratio_j,
        ),
        MetricSpec(
            key="group_ratio_i",
            label="Complementary Group Ratio (i / total)",
            category="ratio",
            description="Complementary group ratio: i-group count divided by total count.",
            formula="(i_tp + i_fp + i_tn + i_fn) / total",
            compute=group_ratio_i,
        ),
        MetricSpec(
            key="imbalance_ratio",
            label="Imbalance Ratio",
            category="ratio",
            description="Positive-class proportion in the full dataset.",
            formula="(i_tp + i_fn + j_tp + j_fn) / total",
            compute=imbalance_ratio,
        ),
        MetricSpec(
            key="accuracy",
            label="Accuracy",
            category="performance",
            description="Overall accuracy over the full confusion matrix.",
            formula="(i_tp + i_tn + j_tp + j_tn) / total",
            compute=accuracy,
        ),
        MetricSpec(
            key="g_mean",
            label="G-mean",
            category="performance",
            description="Geometric mean of the overall true positive rate and true negative rate.",
            formula="sqrt(TPR * TNR)",
            compute=g_mean,
        ),
        MetricSpec(
            key="tpr_i",
            label="TPR (i)",
            category="component",
            description="True positive rate for the i-group.",
            formula="i_tp / (i_tp + i_fn)",
            compute=tpr_i,
        ),
        MetricSpec(
            key="tpr_j",
            label="TPR (j)",
            category="component",
            description="True positive rate for the j-group.",
            formula="j_tp / (j_tp + j_fn)",
            compute=tpr_j,
        ),
        MetricSpec(
            key="fpr_i",
            label="FPR (i)",
            category="component",
            description="False positive rate for the i-group.",
            formula="i_fp / (i_fp + i_tn)",
            compute=fpr_i,
        ),
        MetricSpec(
            key="fpr_j",
            label="FPR (j)",
            category="component",
            description="False positive rate for the j-group.",
            formula="j_fp / (j_fp + j_tn)",
            compute=fpr_j,
        ),
        MetricSpec(
            key="ppv_i",
            label="PPV (i)",
            category="component",
            description="Positive predictive value for the i-group.",
            formula="i_tp / (i_tp + i_fp)",
            compute=ppv_i,
        ),
        MetricSpec(
            key="ppv_j",
            label="PPV (j)",
            category="component",
            description="Positive predictive value for the j-group.",
            formula="j_tp / (j_tp + j_fp)",
            compute=ppv_j,
        ),
        MetricSpec(
            key="npv_i",
            label="NPV (i)",
            category="component",
            description="Negative predictive value for the i-group.",
            formula="i_tn / (i_tn + i_fn)",
            compute=npv_i,
        ),
        MetricSpec(
            key="npv_j",
            label="NPV (j)",
            category="component",
            description="Negative predictive value for the j-group.",
            formula="j_tn / (j_tn + j_fn)",
            compute=npv_j,
        ),
        MetricSpec(
            key="accuracy_equality_difference",
            label="Accuracy Equality Difference",
            category="fairness",
            sort_order=0,
            description="Difference in per-group accuracy, j − i.",
            formula=r"\frac{j_\mathrm{tp}+j_\mathrm{tn}}{n_j} - \frac{i_\mathrm{tp}+i_\mathrm{tn}}{n_i}",
            compute=acc_equality_diff,
        ),
        MetricSpec(
            key="statistical_parity_difference",
            label="Statistical Parity Difference",
            category="fairness",
            sort_order=1,
            description="Difference in predicted-positive rate, j − i.",
            formula=r"\frac{j_\mathrm{tp}+j_\mathrm{fp}}{n_j} - \frac{i_\mathrm{tp}+i_\mathrm{fp}}{n_i}",
            compute=statistical_parity_diff,
        ),
        MetricSpec(
            key="equal_opportunity_difference",
            label="Equal Opportunity Difference",
            category="fairness",
            sort_order=2,
            description="Difference in true positive rate (recall), j − i.",
            formula=r"\frac{j_\mathrm{tp}}{j_\mathrm{tp}+j_\mathrm{fn}} - \frac{i_\mathrm{tp}}{i_\mathrm{tp}+i_\mathrm{fn}}",
            compute=equal_opportunity_diff,
        ),
        MetricSpec(
            key="equalized_odds_difference",
            label="Equalized Odds Difference",
            category="fairness",
            sort_order=3,
            description="Max of |TPR gap| and |FPR gap| between groups. Zero iff equalized odds holds.",
            formula=r"\max\!\bigl(|\Delta\mathrm{TPR}|,\,|\Delta\mathrm{FPR}|\bigr)",
            compute=equalized_odds_diff,
        ),
        MetricSpec(
            key="predictive_equality_difference",
            label="Predictive Equality Difference",
            category="fairness",
            sort_order=4,
            description="Difference in false positive rate, j − i.",
            formula=r"\frac{j_\mathrm{fp}}{j_\mathrm{fp}+j_\mathrm{tn}} - \frac{i_\mathrm{fp}}{i_\mathrm{fp}+i_\mathrm{tn}}",
            compute=predictive_equality_diff,
        ),
        MetricSpec(
            key="positive_predictive_parity_difference",
            label="Positive Predictive Parity Difference",
            category="fairness",
            sort_order=5,
            description="Difference in positive predictive value (precision), j − i.",
            formula=r"\frac{j_\mathrm{tp}}{j_\mathrm{tp}+j_\mathrm{fp}} - \frac{i_\mathrm{tp}}{i_\mathrm{tp}+i_\mathrm{fp}}",
            compute=positive_predictive_parity_diff,
        ),
        MetricSpec(
            key="negative_predictive_parity_difference",
            label="Negative Predictive Parity Difference",
            category="fairness",
            sort_order=6,
            description="Difference in negative predictive value, j − i.",
            formula=r"\frac{j_\mathrm{tn}}{j_\mathrm{tn}+j_\mathrm{fn}} - \frac{i_\mathrm{tn}}{i_\mathrm{tn}+i_\mathrm{fn}}",
            compute=negative_predictive_parity_diff,
        ),
    ]

    for spec in specs:
        register_metric(spec)


register_builtin_metrics()


FAIRNESS_METRIC_KEYS = [
    "accuracy_equality_difference",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "predictive_equality_difference",
    "positive_predictive_parity_difference",
    "negative_predictive_parity_difference",
]

PERFORMANCE_METRIC_KEYS = ["accuracy", "g_mean"]
