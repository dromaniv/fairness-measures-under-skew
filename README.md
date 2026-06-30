# Fairness Measures Under Skew

This project studies how binary-classification fairness measures behave when the
data distribution is skewed. The app compares standard group-fairness measures
with association-based measures under three sources of variation:

- class imbalance, measured by the positive-class ratio (IR);
- protected-group imbalance, measured by the group ratio (GR);
- label-conditional group skew, measured by stereotypical ratios (SR_p, SR_n, SR_c).

The main goal is to make measure behavior visible: when a measure becomes
undefined, when its distribution widens under skew, and when association-based
alternatives stay more stable across comparable settings.

## Measures

The registry includes common fairness differences such as statistical parity,
equal opportunity, predictive equality, equalized odds, positive predictive
parity, negative predictive parity, and accuracy equality. It also includes
association-based measures for 2 x 2 fairness tables, including Fairness Phi,
Marginal Y Association, Conditional Q Association, and Conditional Y Association.

## Workflows

- Synthetic study: enumerate or sample all 8-cell confusion matrices and inspect
  measure distributions, undefined-value rates, and fairness-performance heatmaps.
- Adult case study: resample the UCI Adult dataset at controlled IR and GR values,
  run repeated classifier evaluations, and compare measure stability.
- Stereotypical bias study: compare measure behavior across SR_p, SR_n, and SR_c
  while holding IR and GR fixed.
- Fairness benchmark: inject controlled TPR/FPR discrimination gaps and measure
  each measure's response and detection power.
- Measure registry: browse formulas, descriptions, and categories.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

The Streamlit app is the only entry point.

## Layout

- `app.py`: Streamlit interface.
- `metric_registry.py`: measure registration and safe division.
- `builtin_metrics.py`: standard measures, ratios, and performance measures.
- `custom_metrics.py`: association-based measures and user extensions.
- `synthetic_data.py`: exact and sampled confusion-matrix generation.
- `synthetic_analysis.py`: probability curves and heatmap data.
- `stereotypical_study.py`: SR summaries and measure-by-SR tables.
- `adult_case_study.py`: Adult dataset sampling, preprocessing, and evaluation.
- `fairness_benchmark.py`: controlled discrimination injection.
- `plots.py`: Matplotlib figure builders.
- `data/adult.data`: local Adult dataset file.

## Sources

This project builds on the public code in
[Rasalrai/analysis-of-fairness-measures](https://github.com/Rasalrai/analysis-of-fairness-measures).

The Adult case study uses the
[Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) from the UCI
Machine Learning Repository. The raw files are available in the
[UCI Adult data directory](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/).

## Adding a Measure

Add a function that accepts a `pd.DataFrame` and returns a NumPy array:

```python
def my_measure(df: pd.DataFrame) -> np.ndarray:
    ...
```

Register it with `MetricSpec` in `custom_metrics.py`. Use `safe_divide` for
division and return `np.nan` when a value is undefined. Signed fairness
differences follow the project convention `j - i`.

## Reproducibility

The default random seed is `2137`. Exact enumeration is intended for moderate
sample sizes; large generated confusion-matrix files and exported figures are
excluded from the clean project copy.
