"""
churn_model_selection_toolkit
==============================
Churn Model Selection Toolkit
------------------------------
A curated collection of functions for evaluating, comparing, and selecting
binary classification models in the context of telco churn prediction,
designed for use in Jupyter notebooks.

All tabular outputs are rendered as styled HTML tables using a consistent
visual language designed for readability in academic and professional settings.

Visual Language
---------------
- Centered, bordered HTML tables
- Light grey header background (``#f2f2f2``)
- Bold column headers
- Consistent cell padding
- Gap flags (†) for train–test discrepancies above a configurable threshold

Language Support
----------------
Most functions include a ``spanish`` parameter for bilingual outputs:

    spanish=False → English (default)
    spanish=True  → Spanish

This affects:
- Column headers
- Metric names
- Model labels

Dependencies
------------
    pip install scikit-learn pandas numpy

Functions
---------
Hyperparameter Selection
    display_top5
        Summarises cross-validation results as Median ± IQR and renders
        the top 5 configurations in a styled HTML table.

Model Training & Evaluation
    evaluate_model
        Fits a pipeline on training data and reports train/test performance
        across five standard classification metrics.

Model Comparison
    compare_models_table
        Evaluates multiple fitted pipelines on a shared test set and renders
        a sorted performance table.

Statistical Testing
    compare_delong
        Performs pairwise DeLong tests with Holm–Bonferroni correction across
        multiple fitted models and renders a styled results table.

Notes
-----
- All functions are designed to integrate seamlessly with ``scikit-learn``
  pipelines and ``GridSearchCV`` / ``RandomizedSearchCV`` result DataFrames.
- The shared ``_TABLE_STYLES`` constant ensures visual consistency across
  all tabular outputs.
"""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

__author__ = (
    "Paula Andrea Gómez Vargas <apaulag@uninorte.edu.co>, "
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "display_top5",
    "evaluate_model",
    "compare_models_table",
    "compare_delong",
]

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from itertools import combinations

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML

# ---------------------------------------------------------------------------
# Shared table styling
# ---------------------------------------------------------------------------
_TABLE_STYLES = [
    {
        "selector": "",
        "props": [
            ("margin-left", "auto"),
            ("margin-right", "auto"),
            ("border-collapse", "collapse"),
            ("width", "auto"),
        ],
    },
    {
        "selector": "th",
        "props": [
            ("text-align", "center"),
            ("background-color", "#f2f2f2"),
            ("font-weight", "bold"),
            ("border", "1px solid black"),
            ("padding", "8px"),
        ],
    },
    {
        "selector": "td",
        "props": [
            ("text-align", "center"),
            ("border", "1px solid black"),
            ("padding", "8px"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Hyperparameter Selection
# ---------------------------------------------------------------------------

def display_top5(
    df: pd.DataFrame,
    custom_names: list[str] = None,
) -> None:
    """
    Summarise cross-validation results and display the top 5 configurations.

    Computes the **Median** and **IQR** of per-fold scores from a
    ``GridSearchCV`` or ``RandomizedSearchCV`` result DataFrame, ranks
    configurations by median performance, and renders the top 5 as a
    styled HTML table with scores formatted as ``Median ± IQR``.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame derived from ``estimator.cv_results_``, containing
        columns prefixed with ``split`` (per-fold scores) and ``param_``
        (hyperparameter values).

    custom_names : list of str, optional
        Custom display names for the hyperparameter columns. If provided,
        its length must equal the number of ``param_`` columns found in
        ``df``. When ``None``, automatic prefix stripping is applied:
        ``param_classifier__`` and ``param_`` prefixes are removed.

    Notes
    -----
    **Score aggregation**

    Per-fold scores are identified by ``split``-prefixed columns.
    Median and IQR (Q75 − Q25) are computed row-wise across all folds.

    **Column ordering**

    The rendered table shows hyperparameter columns first, followed by
    the ``F1-Score`` (``Median ± IQR``) column.

    **Numeric formatting**

    Numeric (float) hyperparameter columns are rounded to 3 decimal places.
    The combined ``F1-Score`` column is a formatted string and is not
    subject to numeric rounding.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
    >>> gs.fit(X_train, y_train)
    >>> display_top5(pd.DataFrame(gs.cv_results_))

    >>> display_top5(
    ...     pd.DataFrame(gs.cv_results_),
    ...     custom_names=["Max Depth", "N Estimators"]
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Identify fold and parameter columns
    # ------------------------------------------------------------------
    fold_cols  = [c for c in df.columns if c.startswith("split")]
    param_cols = [c for c in df.columns if c.startswith("param_")]

    # ------------------------------------------------------------------
    # 2. Compute Median and IQR across folds
    # ------------------------------------------------------------------
    df_calc = df.copy()
    df_calc["Median_F1"] = df_calc[fold_cols].median(axis=1)
    df_calc["IQR_F1"] = (
        df_calc[fold_cols].quantile(0.75, axis=1)
        - df_calc[fold_cols].quantile(0.25, axis=1)
    )

    # ------------------------------------------------------------------
    # 3. Extract top 5 by median
    # ------------------------------------------------------------------
    top5 = df_calc.sort_values(by="Median_F1", ascending=False).head(5).copy()

    # ------------------------------------------------------------------
    # 4. Build combined score column
    # ------------------------------------------------------------------
    top5["F1-Score"] = top5.apply(
        lambda row: f"{row['Median_F1']:.3f} ± {row['IQR_F1']:.3f}", axis=1
    )

    # ------------------------------------------------------------------
    # 5. Select and rename columns
    # ------------------------------------------------------------------
    cols_to_show  = param_cols + ["F1-Score"]
    top5_display  = top5[cols_to_show].copy()

    if custom_names and len(custom_names) == len(param_cols):
        top5_display.columns = custom_names + ["F1-Score"]
    else:
        top5_display.columns = [
            c.replace("param_classifier__", "").replace("param_", "")
            if c.startswith("param_") else c
            for c in top5_display.columns
        ]

    # ------------------------------------------------------------------
    # 6. Format and render
    # ------------------------------------------------------------------
    numeric_cols = top5_display.select_dtypes(include=["float", "number"]).columns
    format_dict  = {col: "{:.3f}" for col in numeric_cols}

    styled = (
        top5_display.style
        .format(format_dict)
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='display:flex; justify-content:center; margin-bottom: 20px;'>"
        + styled.to_html()
        + "</div>"
    ))


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alert_threshold: float = 0.03,
    spanish: bool = False,
) -> Pipeline:
    """
    Fit a pipeline and evaluate its performance on both train and test sets.

    Fits the provided ``sklearn.Pipeline`` (with hyperparameters already
    configured) exclusively on training data to prevent data leakage, then
    computes five standard classification metrics on both splits. Metrics
    whose absolute train–test gap exceeds ``alert_threshold`` are flagged
    with a † symbol in the rendered table.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        A fully constructed pipeline (including preprocessing and the final
        estimator) with the desired hyperparameters already set.
        Must expose ``predict`` and ``predict_proba``.

    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training labels (binary 0/1).

    X_test : pd.DataFrame
        Test feature matrix.

    y_test : pd.Series
        Test labels (binary 0/1).

    alert_threshold : float, optional
        Absolute Train–Test difference above which a metric is flagged
        with † in the Gap column (default ``0.03``).

    spanish : bool, optional
        If ``True``, display metric names and column headers in Spanish.
        Default is ``False``.

    Returns
    -------
    pipeline : sklearn.Pipeline
        Fully fitted pipeline ready for inference.

    Notes
    -----
    **Metrics computed**

    +---------------+------------------+
    | English        | Spanish         |
    +===============+==================+
    | Accuracy       | Exactitud       |
    | Precision      | Precisión       |
    | Recall         | Sensibilidad    |
    | F1-Score       | F1-Score        |
    | ROC-AUC        | ROC-AUC         |
    +---------------+------------------+

    **Gap flag (†)**

    The ``Gap`` column shows the absolute difference between train and test
    for each metric. Values exceeding ``alert_threshold`` are appended with
    a † to signal potential overfitting.

    Examples
    --------
    >>> fitted_pipeline = evaluate_model(
    ...     pipeline, X_train, y_train, X_test, y_test,
    ...     alert_threshold=0.05,
    ...     spanish=False
    ... )

    >>> fitted_es = evaluate_model(
    ...     pipeline, X_train, y_train, X_test, y_test,
    ...     spanish=True
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Fit pipeline
    # ------------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 2. Generate predictions
    # ------------------------------------------------------------------
    y_tr_pred = pipeline.predict(X_train)
    y_tr_prob = pipeline.predict_proba(X_train)[:, 1]
    y_te_pred = pipeline.predict(X_test)
    y_te_prob = pipeline.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 3. Language configuration
    # ------------------------------------------------------------------
    if spanish:
        metric_names = ["Exactitud", "Precisión", "Sensibilidad", "F1-Score", "ROC-AUC"]
        col_train    = "Entrenamiento"
        col_test     = "Prueba"
        col_gap      = "Diferencia"
    else:
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        col_train    = "Train"
        col_test     = "Test"
        col_gap      = "Gap"

    metric_fns = [
        (metric_names[0], accuracy_score,  y_tr_pred, y_te_pred),
        (metric_names[1], precision_score, y_tr_pred, y_te_pred),
        (metric_names[2], recall_score,    y_tr_pred, y_te_pred),
        (metric_names[3], f1_score,        y_tr_pred, y_te_pred),
        (metric_names[4], roc_auc_score,   y_tr_prob, y_te_prob),
    ]

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    rows = []
    for name, fn, tr_arg, te_arg in metric_fns:
        tr_val = fn(y_train, tr_arg)
        te_val = fn(y_test,  te_arg)
        rows.append({"Metric": name, col_train: tr_val, col_test: te_val})

    df_summary          = pd.DataFrame(rows)
    df_summary[col_gap] = (df_summary[col_train] - df_summary[col_test]).abs()

    # ------------------------------------------------------------------
    # 5. Gap flagging formatter
    # ------------------------------------------------------------------
    def _flag_gap(val: float) -> str:
        return f"{val:.3f}†" if val > alert_threshold else f"{val:.3f}"

    # ------------------------------------------------------------------
    # 6. Render styled table
    # ------------------------------------------------------------------
    styled = (
        df_summary.style
        .hide(axis="index")
        .format({
            col_train: "{:.3f}",
            col_test:  "{:.3f}",
            col_gap:   _flag_gap,
        })
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))

    return pipeline


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models_table(
    models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    spanish: bool = False,
    sort_by: str = "ROC-AUC",
    tie_breaker: str = "Accuracy",
) -> None:
    """
    Compare multiple fitted pipelines on a shared test set.

    Evaluates each model on five standard classification metrics and renders
    a ranked HTML table. Models are sorted in descending order by
    ``sort_by``, with ``tie_breaker`` applied as a secondary sort key.

    Parameters
    ----------
    models : list of (str, Pipeline)
        Each tuple contains a display name and its corresponding fitted
        ``sklearn.Pipeline``. All pipelines must expose ``predict`` and
        ``predict_proba``.

    X_test : pd.DataFrame
        Test feature matrix shared across all models.

    y_test : pd.Series
        True binary labels (0/1).

    spanish : bool, optional
        If ``True``, display metric names and the model column in Spanish.
        Default is ``False``.

    sort_by : str, optional
        Primary sort metric. Must be one of:
        ``"Accuracy"``, ``"Precision"``, ``"Recall"``,
        ``"F1-Score"``, ``"ROC-AUC"`` (default ``"ROC-AUC"``).

    tie_breaker : str, optional
        Secondary sort metric applied when ``sort_by`` values tie. Must be
        one of the same valid options as ``sort_by``
        (default ``"Accuracy"``).

    Raises
    ------
    ValueError
        If ``sort_by`` or ``tie_breaker`` is not a recognised metric name.

    Notes
    -----
    **Metrics computed**

    +---------------+------------------+
    | English        | Spanish         |
    +===============+==================+
    | Accuracy       | Exactitud       |
    | Precision      | Precisión       |
    | Recall         | Sensibilidad    |
    | F1-Score       | F1-Score        |
    | ROC-AUC        | ROC-AUC         |
    +---------------+------------------+

    **Sorting**

    Both ``sort_by`` and ``tie_breaker`` must be provided in English
    regardless of the ``spanish`` setting; the mapping to Spanish display
    names is handled internally.

    Examples
    --------
    >>> compare_models_table(
    ...     [("LR", lr_pipeline), ("RF", rf_pipeline)],
    ...     X_test, y_test,
    ...     sort_by="F1-Score",
    ...     tie_breaker="ROC-AUC"
    ... )

    >>> compare_models_table(
    ...     [("LR", lr_pipeline), ("RF", rf_pipeline)],
    ...     X_test, y_test,
    ...     spanish=True
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Validate sort keys
    # ------------------------------------------------------------------
    standard_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    if sort_by not in standard_metrics or tie_breaker not in standard_metrics:
        raise ValueError(
            f"sort_by and tie_breaker must be one of: {standard_metrics}"
        )

    # ------------------------------------------------------------------
    # 2. Language mapping
    # ------------------------------------------------------------------
    es_map = {
        "Accuracy":  "Exactitud",
        "Precision": "Precisión",
        "Recall":    "Sensibilidad",
        "F1-Score":  "F1-Score",
        "ROC-AUC":   "ROC-AUC",
    }

    if spanish:
        metric_names = [es_map[m] for m in standard_metrics]
        model_col    = "Modelo"
        sort_col_1   = es_map[sort_by]
        sort_col_2   = es_map[tie_breaker]
    else:
        metric_names = standard_metrics
        model_col    = "Model"
        sort_col_1   = sort_by
        sort_col_2   = tie_breaker

    # ------------------------------------------------------------------
    # 3. Evaluate each model
    # ------------------------------------------------------------------
    rows = []
    for name, pipeline in models:
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        rows.append({
            model_col:       name,
            metric_names[0]: accuracy_score(y_test, y_pred),
            metric_names[1]: precision_score(y_test, y_pred),
            metric_names[2]: recall_score(y_test, y_pred),
            metric_names[3]: f1_score(y_test, y_pred),
            metric_names[4]: roc_auc_score(y_test, y_prob),
        })

    # ------------------------------------------------------------------
    # 4. Sort and render
    # ------------------------------------------------------------------
    df_results = (
        pd.DataFrame(rows)
        .sort_values(by=[sort_col_1, sort_col_2], ascending=[False, False])
    )

    styled = (
        df_results.style
        .format({col: "{:.3f}" for col in df_results.columns if col != model_col})
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))


# ---------------------------------------------------------------------------
# Statistical Testing — DeLong
# ---------------------------------------------------------------------------

def _compute_delong_cov(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the standard error and Z-statistic for two AUCs via DeLong's method.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1).

    y_prob_1 : np.ndarray
        Predicted probabilities from the first model.

    y_prob_2 : np.ndarray
        Predicted probabilities from the second model.

    Returns
    -------
    se : float
        Standard error of the AUC difference.

    z : float
        Z-statistic for the hypothesis test H₀: AUC₁ = AUC₂.
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m, n    = len(pos_idx), len(neg_idx)

    def _v10(probs):
        return np.array([
            np.sum(probs[pos_idx] > probs[j])
            + 0.5 * np.sum(probs[pos_idx] == probs[j])
            for j in neg_idx
        ]) / m

    def _v01(probs):
        return np.array([
            np.sum(probs[i] > probs[neg_idx])
            + 0.5 * np.sum(probs[i] == probs[neg_idx])
            for i in pos_idx
        ]) / n

    v10_1, v01_1 = _v10(y_prob_1), _v01(y_prob_1)
    v10_2, v01_2 = _v10(y_prob_2), _v01(y_prob_2)

    s10    = np.cov(v10_1, v10_2)[0, 1]
    s01    = np.cov(v01_1, v01_2)[0, 1]
    cov_12 = s10 / n + s01 / m

    var_diff = (
        np.var(v10_1) / n + np.var(v01_1) / m
        + np.var(v10_2) / n + np.var(v01_2) / m
        - 2 * cov_12
    )

    se   = np.sqrt(np.maximum(var_diff, 1e-8))
    diff = roc_auc_score(y_true, y_prob_1) - roc_auc_score(y_true, y_prob_2)
    z    = diff / se

    return se, z


def compare_delong(
    models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    spanish: bool = False,
    show_p: bool = False,
) -> None:
    """
    Pairwise DeLong test with Holm–Bonferroni correction across multiple models.

    For every pair of fitted pipelines, the DeLong method is used to test
    whether their ROC-AUC scores differ significantly. Bootstrap confidence
    intervals are computed for the AUC difference. Multiple-comparison
    correction is applied via the Holm–Bonferroni procedure. Results are
    rendered as a styled HTML table sorted by absolute AUC difference.

    Parameters
    ----------
    models : list of (str, Pipeline)
        Each tuple contains a display name and its corresponding fitted
        ``sklearn.Pipeline``. All pipelines must expose ``predict_proba``.

    X_test : pd.DataFrame
        Test feature matrix shared across all models.

    y_test : pd.Series
        True binary labels (0/1).

    n_bootstraps : int, optional
        Number of bootstrap resamples used to compute the confidence
        interval for the AUC difference (default ``2000``).

    alpha : float, optional
        Significance level for both multiple-comparison correction and
        bootstrap confidence intervals (default ``0.05``).

    spanish : bool, optional
        If ``True``, rename ``Model 1`` / ``Model 2`` columns to
        ``Modelo 1`` / ``Modelo 2`` and relabel ``Difference`` as
        ``Diferencia``. Default is ``False``.

    show_p : bool, optional
        If ``True``, a dedicated ``p`` column (Holm-corrected, formatted
        with significance stars) is added. If ``False`` (default),
        significance stars are appended directly to the ``Z`` column and
        the ``p`` column is omitted.

    Notes
    -----
    **Significance stars**

    +--------+----------------------------+
    | Symbol | Condition (corrected p)    |
    +========+============================+
    | ***    | p < 0.001                  |
    | **     | p < 0.01                   |
    | *      | p < 0.05                   |
    | (none) | p ≥ 0.05                   |
    +--------+----------------------------+

    **Confidence intervals**

    Bootstrap CIs are computed at level ``1 − alpha`` using percentile
    resampling. Bootstrap rounds where ``y_true_boot`` contains only one
    class are silently skipped.

    **Table columns**

    ``Model 1``, ``Model 2``, ``Difference``, ``SE``, ``Z`` (or ``Z*``),
    ``LB``, ``UB`` — and optionally ``p`` when ``show_p=True``.

    Examples
    --------
    >>> compare_delong(
    ...     [("LR", lr_pipeline), ("RF", rf_pipeline)],
    ...     X_test, y_test,
    ...     n_bootstraps=1000,
    ...     show_p=True
    ... )

    >>> compare_delong(
    ...     [("LR", lr_pipeline), ("RF", rf_pipeline)],
    ...     X_test, y_test,
    ...     spanish=True,
    ...     alpha=0.01
    ... )
    """
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test

    # ------------------------------------------------------------------
    # 1. Collect probabilities and AUCs
    # ------------------------------------------------------------------
    probs = {}
    aucs  = {}
    for name, pipeline in models:
        y_prob        = pipeline.predict_proba(X_test)[:, 1]
        probs[name]   = y_prob
        aucs[name]    = roc_auc_score(y_test_arr, y_prob)

    # ------------------------------------------------------------------
    # 2. Pairwise tests
    # ------------------------------------------------------------------
    rows        = []
    p_values_raw = []

    for name1, name2 in combinations(probs.keys(), 2):
        prob1 = probs[name1]
        prob2 = probs[name2]

        diff     = aucs[name1] - aucs[name2]
        se, z    = _compute_delong_cov(y_test_arr, prob1, prob2)
        p_val    = 2 * stats.norm.sf(np.abs(z))
        p_values_raw.append(p_val)

        boot_diffs = []
        for _ in range(n_bootstraps):
            indices    = resample(np.arange(len(y_test_arr)))
            y_true_b   = y_test_arr[indices]
            if len(np.unique(y_true_b)) < 2:
                continue
            boot_diffs.append(
                roc_auc_score(y_true_b, prob1[indices])
                - roc_auc_score(y_true_b, prob2[indices])
            )

        lb = np.percentile(boot_diffs, (alpha / 2) * 100)
        ub = np.percentile(boot_diffs, (1 - alpha / 2) * 100)

        rows.append({
            "Model 1":    name1,
            "Model 2":    name2,
            "Difference": diff,
            "SE":         se,
            "Z":          z,
            "LB":         lb,
            "UB":         ub,
        })

    df_results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Multiple-comparison correction
    # ------------------------------------------------------------------
    _, pvals_corrected, _, _ = multipletests(
        p_values_raw, alpha=alpha, method="holm"
    )

    formatted_p = []
    stars_list  = []
    for p in pvals_corrected:
        if p < 0.001:
            stars_list.append("***")
            formatted_p.append("<0.001***")
        elif p < 0.01:
            stars_list.append("**")
            formatted_p.append(f"{p:.3f}**")
        elif p < 0.05:
            stars_list.append("*")
            formatted_p.append(f"{p:.3f}*")
        else:
            stars_list.append("")
            formatted_p.append(f"{p:.3f}")

    # ------------------------------------------------------------------
    # 4. Attach p-value or embed stars in Z
    # ------------------------------------------------------------------
    if show_p:
        df_results.insert(5, "p", formatted_p)
    else:
        df_results["Z"] = [
            f"{z:.3f}{stars}"
            for z, stars in zip(df_results["Z"], stars_list)
        ]

    # ------------------------------------------------------------------
    # 5. Language renaming and sorting
    # ------------------------------------------------------------------
    if spanish:
        df_results = df_results.rename(columns={
            "Model 1":    "Modelo 1",
            "Model 2":    "Modelo 2",
            "Difference": "Diferencia",
        })

    sort_col = "Diferencia" if spanish else "Difference"
    df_results = df_results.sort_values(by=sort_col, ascending=False, key=abs)

    # ------------------------------------------------------------------
    # 6. Render styled table
    # ------------------------------------------------------------------
    exclude_cols = (
        ["Model 1", "Model 2", "Modelo 1", "Modelo 2"]
        + (["p"] if show_p else ["Z"])
    )
    numeric_cols = [c for c in df_results.columns if c not in exclude_cols]

    styled = (
        df_results.style
        .format({col: "{:.3f}" for col in numeric_cols})
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
    )

    display(HTML(
        "<div style='text-align: center; width: 100%; margin-top: 10px;'>"
        + styled.to_html()
        + "</div>"
    ))