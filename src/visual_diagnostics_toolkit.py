"""
visual_diagnostics_toolkit
=========================
Model Visual Diagnostics & Interpretation Toolkit
-------------------------------------------------
A curated collection of functions for interpreting and visualising
binary classification models in Jupyter notebooks.

All functions produce publication-ready figures rendered as inline
Base64-encoded images using a consistent visual language aligned with
``model_evaluation_toolkit.py``.

Visual Language
---------------
- Clean white background (seaborn "white")
- Bold axis labels and titles
- Black axis borders
- Subtle dashed gridlines
- Centered rendering via HTML
- Default output resolution: 100–120 dpi

Language Support
----------------
Most functions include a ``spanish`` parameter for bilingual outputs:

    spanish=False → English (default)
    spanish=True  → Spanish

This affects:
- Plot titles
- Axis labels
- Class labels

Dependencies
------------
    pip install scikit-learn pandas numpy matplotlib seaborn lime

Functions
---------
Feature Importance
    plot_permutation_importance
        Permutation-based feature importance with 95% confidence intervals.

Local Explanations
    plot_lime_explanation
        Local feature importance for a single instance using LIME.

Diagnostics
    plot_confusion_matrix
        Styled confusion matrix with full control over aesthetics.

Notes
-----
- All visual outputs are optimised for **academic publication**.
- Designed to integrate seamlessly with the modeling pipeline.

Changelog
---------
v2.0.0
    - Added ``plot_lime_explanation`` for per-instance LIME-based explanations.
    - Added ``_SECONDARY_RED`` global colour constant for negative LIME weights.
    - Expanded ``__all__`` to include the new function.
"""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "2.0.0"

__author__ = (
    "Paula Andrea Gómez Vargas <apaulag@uninorte.edu.co>, "
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "plot_permutation_importance",
    "plot_lime_explanation",
    "plot_confusion_matrix",
]


import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
_PRIMARY_BLUE   = "#4C72B0"
_SECONDARY_RED  = "#C44E52"


def _apply_global_style() -> None:
    """Apply consistent rcParams and seaborn styling."""
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5
    })
    sns.set_style("white")


def _render_figure(fig: plt.Figure, dpi: int = 100) -> None:
    """
    Encode a matplotlib figure as Base64 PNG and render it centered.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    dpi : int, optional
        Output resolution (default 100).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    encoded = base64.b64encode(buf.getbuffer()).decode("ascii")

    display(HTML(
        f'<div style="text-align: center; width: 100%;">'
        f'<img src="data:image/png;base64,{encoded}"></div>'
    ))


# ---------------------------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------------------------

def plot_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 200,
    scoring: str = "roc_auc",
    top_k: int = 10,
    interval_type: str = "bootstrap",
    title: str = None,
    label_pad: int = 15,
    xlim: tuple = None,
    tick_size: int = 12,
    annot_size: int = 12,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 5),
    spanish: bool = True,
) -> None:
    """
    Plot permutation feature importance with statistical filtering and full styling control.

    This function computes permutation-based feature importance and displays
    only statistically significant variables based on confidence intervals.
    It supports both bootstrap percentile intervals and normal approximation.

    A horizontal bar chart is produced with asymmetric error bars and
    annotated mean importance values.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model supporting ``predict`` or ``predict_proba``.

    X : pd.DataFrame
        Feature matrix used for evaluation.

    y : pd.Series
        True labels.

    n_repeats : int, optional
        Number of permutations per feature (default ``200``).

    scoring : str, optional
        Metric used to evaluate performance degradation
        (default ``"roc_auc"``).

    top_k : int, optional
        Number of top features to display after filtering
        (default ``10``).

    interval_type : {"bootstrap", "normal"}, optional
        Type of confidence interval:
            - "bootstrap" → percentile-based (recommended)
            - "normal"    → mean ± 1.96·SE

    title : str, optional
        Custom plot title.

    label_pad : int, optional
        Padding for axis labels.

    xlim : tuple, optional
        Manual limits for the x-axis.

    tick_size : int, optional
        Font size for axis ticks.

    annot_size : int, optional
        Font size for numeric annotations.

    random_state : int, optional
        Random seed for reproducibility.

    figsize : tuple, optional
        Figure size in inches.

    spanish : bool, optional
        If True, display labels in Spanish.
        Default is ``True``.

    Notes
    -----
    **Statistical filtering**

    Only features whose confidence interval does **not include zero**
    are displayed. This ensures that all plotted variables have a
    statistically significant impact on model performance.

    **Interpretation**

    - Larger values → greater importance
    - Error bars → uncertainty in importance estimates
    - If no features pass the filter, a warning is displayed

    **Interval types**

    - Bootstrap (default): robust, non-parametric
    - Normal: faster but assumes approximate normality

    **Axis scaling**

    If ``xlim`` is not provided, limits are automatically adjusted
    to leave space for annotations.

    Examples
    --------
    >>> plot_permutation_importance(
    ...     model, X_test, y_test,
    ...     scoring="roc_auc",
    ...     interval_type="bootstrap",
    ...     top_k=10,
    ...     spanish=True
    ... )

    >>> plot_permutation_importance(
    ...     model, X_test, y_test,
    ...     interval_type="normal",
    ...     xlim=(0, 0.2),
    ...     tick_size=14,
    ...     annot_size=13
    ... )
    """
    from sklearn.inspection import permutation_importance

    # ------------------------------------------------------------------
    # 1. Compute permutation importance
    # ------------------------------------------------------------------
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring
    )

    # ------------------------------------------------------------------
    # 2. Confidence intervals
    # ------------------------------------------------------------------
    mean = result.importances_mean

    if interval_type == "bootstrap":
        lower = np.percentile(result.importances, 2.5, axis=1)
        upper = np.percentile(result.importances, 97.5, axis=1)
        err_l, err_h = mean - lower, upper - mean
    else:
        se = result.importances_std
        lower, upper = mean - (1.96 * se), mean + (1.96 * se)
        err_l = err_h = 1.96 * se

    df_all = pd.DataFrame({
        "Feature": X.columns,
        "Mean": mean,
        "Lower": lower,
        "Err_L": err_l,
        "Err_H": err_h
    })

    # ------------------------------------------------------------------
    # 3. Statistical significance filtering
    # ------------------------------------------------------------------
    df_sig = (
        df_all[df_all["Lower"] > 0]
        .sort_values(by="Mean", ascending=True)
        .tail(top_k)
    )

    if df_sig.empty:
        print(
            "Aviso: Ninguna variable superó el filtro de significancia estadística."
            if spanish else
            "Warning: No features passed the statistical significance filter."
        )
        return

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    xerr = [df_sig["Err_L"].values, df_sig["Err_H"].values]

    bars = ax.barh(
        df_sig["Feature"],
        df_sig["Mean"],
        xerr=xerr,
        color=_PRIMARY_BLUE,
        edgecolor="black",
        alpha=0.8,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"}
    )

    # ------------------------------------------------------------------
    # 5. Labels and axes
    # ------------------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=tick_size + 4, fontweight="bold", pad=25)

    xlabel = (
        f"Caída en {scoring.upper()}"
        if spanish else
        f"Decrease in {scoring.upper()}"
    )

    ylabel = "Atributos" if spanish else "Features"

    ax.set_xlabel(xlabel, fontsize=tick_size + 2, fontweight="bold", labelpad=label_pad)
    ax.set_ylabel(ylabel, fontsize=tick_size + 2, fontweight="bold", labelpad=label_pad)

    ax.tick_params(axis="both", labelsize=tick_size)

    # ------------------------------------------------------------------
    # 6. Axis limits
    # ------------------------------------------------------------------
    if xlim:
        ax.set_xlim(xlim)
    else:
        max_visible = (df_sig["Mean"] + df_sig["Err_H"]).max()
        ax.set_xlim(0, max_visible * 1.15)

    # ------------------------------------------------------------------
    # 7. Annotations
    # ------------------------------------------------------------------
    offset = ax.get_xlim()[1] * 0.01

    for i, bar in enumerate(bars):
        val_mean = df_sig["Mean"].iloc[i]
        val_err_h = df_sig["Err_H"].iloc[i]

        ax.text(
            val_mean + val_err_h + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val_mean:.3f}",
            va="center",
            ha="left",
            fontsize=annot_size,
            fontweight="bold",
            color="#2c3e50"
        )

    # ------------------------------------------------------------------
    # 8. Styling
    # ------------------------------------------------------------------
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    _render_figure(fig, dpi=120)


# ---------------------------------------------------------------------------
# LIME Local Explanation
# ---------------------------------------------------------------------------

def plot_lime_explanation(
    explainer,
    instance: "pd.Series | np.ndarray",
    pipeline,
    true_label: "str | int" = "N/A",
    target_labels: dict = None,
    note_pos: str = "upper right",
    label_encoders: dict = None,
    feature_labels: "dict[str, str]" = None,
    top_k: int = 5,
    label_pad: int = 15,
    xlim: tuple = None,
    tick_size: int = 12,
    annot_size: int = 12,
    figsize: "tuple[int, int]" = (8, 5),
    spanish: bool = True,
) -> None:
    """
    Plot local feature importance for a single instance using LIME.

    Generates a horizontal bar chart where each bar represents the
    contribution of a feature to the model's prediction for the given
    instance. Positive weights push the prediction toward the positive
    class; negative weights push it toward the negative class.

    The predicted class and its probability (displayed between 0 and 1
    with 3 decimal places) are shown in a metadata box inside the plot.

    Parameters
    ----------
    explainer : lime.lime_tabular.LimeTabularExplainer
        A fitted LIME explainer instance.

    instance : pd.Series or np.ndarray
        The single observation to explain.
        If a ``pd.Series``, column names are used for feature lookup and
        optional inverse-encoding via ``label_encoders``.

    pipeline : sklearn Pipeline or estimator
        A fitted model (or pipeline) exposing ``predict_proba``.

    true_label : str or int, optional
        The ground-truth label for this instance (default ``"N/A"``).
        Displayed in the metadata box for reference.

    target_labels : dict, optional
        Mapping from raw target values to human-readable strings,
        e.g. ``{0: "No Disease", 1: "Disease"}``.
        Applied to ``true_label`` before display.

    note_pos : {"upper right", "upper left"}, optional
        Position of the metadata annotation box (default ``"upper right"``).

    label_encoders : dict, optional
        Mapping ``{column_name: LabelEncoder}`` used to inverse-transform
        integer-encoded categorical columns before passing to ``pipeline``.
        Only relevant when ``instance`` is a ``pd.Series``.

    feature_labels : dict[str, str], optional
        Mapping of raw feature name substrings to display-friendly labels,
        e.g. ``{"age_years": "Age (years)"}``.
        Applied via substring replacement after pipeline-prefix cleaning.

    top_k : int, optional
        Number of top features to display (default ``5``).

    label_pad : int, optional
        Padding for axis labels (default ``15``).

    xlim : tuple, optional
        Manual limits for the x-axis. If ``None``, limits are set
        symmetrically based on the maximum absolute weight.

    tick_size : int, optional
        Font size for axis ticks (default ``12``).

    annot_size : int, optional
        Font size for bar annotations (default ``12``).

    figsize : tuple[int, int], optional
        Figure size in inches (default ``(8, 5)``).

    spanish : bool, optional
        If ``True``, display axis labels and metadata in Spanish.
        Default is ``True``.

    Notes
    -----
    **Colour coding**

    - Blue (``_PRIMARY_BLUE``) → positive weight (supports predicted class)
    - Red  (``_SECONDARY_RED``) → negative weight (opposes predicted class)

    **Feature name cleaning**

    Pipeline-prefixed names (e.g. ``"preprocessor__age"``) are
    automatically stripped to their base name (``"age"``).
    Additional renaming is applied via ``feature_labels``.

    **Predict wrapper**

    An internal ``safe_predict_fn`` handles the conversion from raw
    NumPy arrays back to the expected DataFrame format, including
    inverse-encoding of categorical columns, before calling
    ``pipeline.predict_proba``.

    **Probability display**

    Confidence is shown as a probability in ``[0, 1]`` with 3 decimal
    places, regardless of the LIME explainer's internal scaling.

    Examples
    --------
    >>> plot_lime_explanation(
    ...     explainer, X_test.iloc[0], pipeline,
    ...     true_label=y_test.iloc[0],
    ...     target_labels={0: "Sin EC", 1: "Con EC"},
    ...     top_k=8,
    ...     spanish=True
    ... )

    >>> plot_lime_explanation(
    ...     explainer, X_test.iloc[5], pipeline,
    ...     true_label=y_test.iloc[5],
    ...     note_pos="upper left",
    ...     xlim=(-0.3, 0.3),
    ...     tick_size=13,
    ...     spanish=False
    ... )
    """

    # ------------------------------------------------------------------
    # 0. Safe Predict Wrapper
    # ------------------------------------------------------------------
    def safe_predict_fn(X_array):
        if isinstance(instance, pd.Series):
            X_df = pd.DataFrame(X_array, columns=instance.index)

            if label_encoders is not None:
                for col_name, le in label_encoders.items():
                    if col_name in X_df.columns:
                        X_df[col_name] = le.inverse_transform(
                            X_df[col_name].astype(int)
                        )

            return pipeline.predict_proba(X_df)

        return pipeline.predict_proba(X_array)

    # ------------------------------------------------------------------
    # 1. Prediction Info & LIME Explanation
    # ------------------------------------------------------------------
    inst_input = (
        instance.values.reshape(1, -1)
        if isinstance(instance, pd.Series)
        else instance.reshape(1, -1)
    )

    probs     = safe_predict_fn(inst_input)[0]
    pred_idx  = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])          # probability in [0, 1]

    class_names = (
        explainer.class_names
        if hasattr(explainer, "class_names") and explainer.class_names is not None
        else [0, 1]
    )

    pred_label = class_names[pred_idx]

    mapped_true_label = (
        target_labels.get(true_label, true_label)
        if target_labels
        else true_label
    )

    # LIME explanation
    exp = explainer.explain_instance(
        instance if isinstance(instance, np.ndarray) else instance.values,
        safe_predict_fn,
        num_features=top_k,
    )

    lime_list = exp.as_list()

    if not lime_list:
        msg = (
            "Aviso: LIME no generó ninguna explicación para esta instancia."
            if spanish else
            "Warning: LIME produced no explanation for this instance."
        )
        print(msg)
        return

    df_lime = (
        pd.DataFrame(lime_list, columns=["Feature", "Weight"])
        .iloc[::-1]
        .reset_index(drop=True)
        .copy()
    )

    # ------------------------------------------------------------------
    # 2. Feature Name Cleaning & Custom Labels
    # ------------------------------------------------------------------
    df_lime["Feature"] = df_lime["Feature"].apply(
        lambda x: x.split("__")[-1] if "__" in x else x
    )

    if feature_labels:
        for old, new in feature_labels.items():
            df_lime["Feature"] = df_lime["Feature"].str.replace(
                old, new, regex=False
            )

    # ------------------------------------------------------------------
    # 3. Plot Setup
    # ------------------------------------------------------------------
    _apply_global_style()

    colors = [
        _PRIMARY_BLUE if w > 0 else _SECONDARY_RED
        for w in df_lime["Weight"]
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    bars = ax.barh(
        df_lime["Feature"],
        df_lime["Weight"],
        color=colors,
        edgecolor="black",
        alpha=0.8
    )

    # ------------------------------------------------------------------
    # 4. Axis Labels
    # ------------------------------------------------------------------
    ax.set_xlabel(
        "Peso (Impacto)" if spanish else "Weight (Impact)",
        fontsize=tick_size + 2,
        fontweight="bold",
        labelpad=label_pad
    )

    ax.set_ylabel(
        "Atributos" if spanish else "Features",
        fontsize=tick_size + 2,
        fontweight="bold",
        labelpad=label_pad
    )

    ax.tick_params(axis="both", labelsize=tick_size)

    ax.axvline(0, color="black", linewidth=1.5, alpha=0.7)

    # ------------------------------------------------------------------
    # 5. Axis Limits
    # ------------------------------------------------------------------
    if xlim:
        ax.set_xlim(xlim)
    else:
        m = df_lime["Weight"].abs().max()
        ax.set_xlim(-m * 1.5, m * 1.5)

    # ------------------------------------------------------------------
    # 6. Bar Annotations
    # ------------------------------------------------------------------
    off = ax.get_xlim()[1] * 0.02

    for i, bar in enumerate(bars):
        val = df_lime["Weight"].iloc[i]

        ax.text(
            val + off if val > 0 else val - off,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=annot_size,
            fontweight="bold",
            color="#2c3e50"
        )

    # ------------------------------------------------------------------
    # 7. Metadata Box
    # ------------------------------------------------------------------
    info = (
        f"Predicción: {pred_label} ({pred_conf:.3f})\n"
        f"Etiqueta Real: {mapped_true_label}"
        if spanish else
        f"Prediction: {pred_label} ({pred_conf:.3f})\n"
        f"True Label: {mapped_true_label}"
    )

    if note_pos == "upper left":
        x_text, ha_text = 0.03, "left"
    else:
        x_text, ha_text = 0.97, "right"

    ax.text(
        x_text,
        0.95,
        info,
        transform=ax.transAxes,
        fontsize=tick_size - 1,
        va="top",
        ha=ha_text,
        fontweight="bold",
        color="#2c3e50",
        bbox=dict(facecolor="white", alpha=0.0, edgecolor="none")
    )

    # ------------------------------------------------------------------
    # 8. Final Styling
    # ------------------------------------------------------------------
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    _render_figure(fig, dpi=120)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    labels: "list | None" = None,
    title: "str | None" = None,
    figsize: "tuple[int, int]" = (8, 6),
    cmap: str = "Blues",
    label_size: int = 12,
    label_pad: int = 15,
    annot_size: int = 14,
    spanish: bool = True,
) -> None:
    """
    Render a styled confusion matrix with full aesthetic control.

    This function produces a publication-ready confusion matrix using
    ``sklearn.metrics.ConfusionMatrixDisplay`` with enhanced styling.

    Parameters
    ----------
    model : sklearn estimator
        A fitted classification model.

    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        True labels.

    labels : list, optional
        Class labels for display.

    title : str, optional
        Plot title.

    figsize : tuple, optional
        Figure size in inches.

    cmap : str, optional
        Colormap used for the matrix (default ``"Blues"``).

    label_size : int, optional
        Font size for axis labels.

    label_pad : int, optional
        Padding for axis labels.

    annot_size : int, optional
        Font size for cell annotations.

    spanish : bool, optional
        If True, display labels in Spanish.
        Default is ``True``.

    Notes
    -----
    Default labels:
        Spanish → ["Sin EC", "Con EC"]
        English → ["No Disease", "Disease"]

    Examples
    --------
    >>> plot_confusion_matrix(
    ...     model, X_test, y_test,
    ...     spanish=True
    ... )
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    if labels is None:
        labels = (
            ["Sin EC", "Con EC"]
            if spanish else
            ["No Disease", "Disease"]
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, values_format="d", colorbar=False)

    if title is None:
        title = "Matriz de Confusión" if spanish else "Confusion Matrix"

    xlabel = "Etiqueta Predicha" if spanish else "Predicted Label"
    ylabel = "Etiqueta Real"     if spanish else "True Label"

    ax.set_title(title,   fontsize=label_size + 2, fontweight="bold", pad=label_pad - 2)
    ax.set_xlabel(xlabel, fontsize=label_size,     fontweight="bold", labelpad=label_pad)
    ax.set_ylabel(ylabel, fontsize=label_size,     fontweight="bold", labelpad=label_pad)

    for text in disp.text_.ravel():
        text.set_fontsize(annot_size)
        text.set_fontweight("bold")

    ax.tick_params(axis="both", labelsize=label_size - 1)

    plt.tight_layout()
    _render_figure(fig, dpi=150)