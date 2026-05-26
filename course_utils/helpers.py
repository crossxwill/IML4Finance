"""
Helper functions and classes for IML4Finance labs.

This module contains common utilities used across multiple labs including:
- AutoGluon sklearn wrapper
- Random seed setting
- Model folder management
- Plotting utilities
- Scoring and evaluation functions
"""

# System utilities
import os
import shutil
import random
import warnings
import time
import gc
import re
from typing import Optional, List, Dict, Any, Tuple, Callable

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, special
from sklearn.feature_selection import mutual_info_classif

# Machine learning - scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Specialized ML libraries
from autogluon.tabular import TabularPredictor, TabularDataset
import torch


# =============================================================================
# RANDOM SEED UTILITIES
# =============================================================================

def global_set_seed(seed_value: int = 2025) -> None:
    """
    Sets random seeds for reproducibility across multiple libraries.
    
    Parameters
    ----------
    seed_value : int, default=2025
        The seed value to use for all random number generators.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    



# =============================================================================
# FILE SYSTEM UTILITIES
# =============================================================================

def remove_ag_folder(mdl_folder: str) -> None:
    """
    Removes the AutoGluon model folder if it exists.
    
    Parameters
    ----------
    mdl_folder : str
        Path to the AutoGluon model folder to remove.
    """
    if os.path.exists(mdl_folder):
        shutil.rmtree(mdl_folder)
        print(f"Removed existing AutoGluon folder: {mdl_folder}")


def select_top_non_ensemble_models(
    leaderboard: pd.DataFrame,
    n: int = 1,
    ensemble_prefix: str = "WeightedEnsemble",
) -> list[str]:
    """
    Returns the top-ranked non-weighted-ensemble model names from a leaderboard.

    Parameters
    ----------
    leaderboard : pandas.DataFrame
        AutoGluon leaderboard sorted in rank order.

    n : int, default=1
        Number of top non-weighted-ensemble model names to return.

    ensemble_prefix : str, default="WeightedEnsemble"
        Prefix used by AutoGluon weighted ensemble model names.

    Returns
    -------
    list of str
        Top-ranked non-weighted-ensemble model names.

    Raises
    ------
    ValueError
        If the leaderboard is missing a model column, contains only ensembles,
        or n is less than 1.
    """
    if "model" not in leaderboard.columns:
        raise ValueError("Leaderboard must contain a 'model' column.")
    if n < 1:
        raise ValueError("n must be at least 1.")

    model_names = leaderboard["model"].astype(str)
    non_ensemble_models = model_names[
        ~model_names.str.startswith(ensemble_prefix, na=False)
    ]
    top_models = non_ensemble_models.head(n).tolist()

    if not top_models:
        raise ValueError("Leaderboard does not contain a non-weighted-ensemble model.")

    return top_models


def select_best_non_ensemble_model(
    leaderboard: pd.DataFrame,
    ensemble_prefix: str = "WeightedEnsemble",
) -> str:
    """
    Returns the highest-ranked non-weighted-ensemble model from a leaderboard.

    Parameters
    ----------
    leaderboard : pandas.DataFrame
        AutoGluon leaderboard sorted in rank order.

    ensemble_prefix : str, default="WeightedEnsemble"
        Prefix used by AutoGluon weighted ensemble model names.

    Returns
    -------
    str
        Model name for the best non-weighted-ensemble row.

    Raises
    ------
    ValueError
        If the leaderboard is missing a model column or contains only ensembles.
    """
    return select_top_non_ensemble_models(
        leaderboard=leaderboard,
        n=1,
        ensemble_prefix=ensemble_prefix,
    )[0]


# =============================================================================
# LAB 04 SHARED UTILITIES
# =============================================================================

def extract_positive_class_scores(probabilities) -> np.ndarray:
    """
    Extract positive-class scores from common probability outputs.

    Parameters
    ----------
    probabilities : array-like, pandas.Series, or pandas.DataFrame
        Probability output from a binary classifier.

    Returns
    -------
    np.ndarray
        One-dimensional array of positive-class scores.
    """
    if isinstance(probabilities, pd.DataFrame):
        if probabilities.shape[1] == 0:
            raise ValueError("probabilities must contain at least one column.")
        if probabilities.shape[1] == 1:
            return probabilities.iloc[:, 0].to_numpy(dtype=float)
        for positive_column in (1, "1", True):
            if positive_column in probabilities.columns:
                return probabilities[positive_column].to_numpy(dtype=float)
        return probabilities.iloc[:, -1].to_numpy(dtype=float)

    if isinstance(probabilities, pd.Series):
        return probabilities.to_numpy(dtype=float)

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 0:
        return probabilities.reshape(1)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be one- or two-dimensional.")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one column.")

    return probabilities[:, -1]


def _resolve_autogluon_predictor(estimator):
    """Return an underlying AutoGluon predictor when one is available."""
    if isinstance(estimator, TabularPredictor):
        return estimator

    predictor = getattr(estimator, "predictor_", None)
    if isinstance(predictor, TabularPredictor):
        return predictor

    predictor = getattr(estimator, "predictor", None)
    if isinstance(predictor, TabularPredictor):
        return predictor

    if hasattr(estimator, "steps"):
        for _, step in reversed(estimator.steps):
            predictor = _resolve_autogluon_predictor(step)
            if predictor is not None:
                return predictor

    return None


def _coerce_autogluon_prediction_features(estimator, predictor, features):
    """Coerce prediction features into a DataFrame/TabularDataset for AutoGluon."""
    if isinstance(features, TabularDataset):
        return features

    if hasattr(estimator, "_validate_features"):
        try:
            return TabularDataset(estimator._validate_features(features))
        except Exception:
            pass

    if hasattr(features, "columns"):
        return TabularDataset(features)

    feature_names = getattr(estimator, "feature_names_", None)
    if feature_names is None and hasattr(predictor, "feature_metadata_in"):
        feature_names = predictor.feature_metadata_in.get_features()

    feature_array = np.asarray(features)
    if feature_array.ndim != 2:
        raise ValueError(
            "features must be two-dimensional when model_name is provided "
            "for AutoGluon predictions."
        )
    if feature_names is None:
        raise ValueError(
            "Could not infer feature names for the requested AutoGluon model prediction."
        )

    return TabularDataset(pd.DataFrame(feature_array, columns=list(feature_names)))


def predict_positive_class_scores(
    estimator,
    features,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Predict positive-class scores from a fitted estimator or predictor.

    Parameters
    ----------
    estimator : object
        Fitted estimator with predict_proba, TabularPredictor, or an object that
        exposes an underlying AutoGluon predictor.

    features : array-like or DataFrame
        Feature matrix used for prediction.

    model_name : str, optional
        AutoGluon model name to score when the estimator exposes a
        TabularPredictor.

    Returns
    -------
    np.ndarray
        One-dimensional array of positive-class scores.
    """
    if model_name is not None:
        predictor = _resolve_autogluon_predictor(estimator)
        if predictor is not None:
            ag_features = _coerce_autogluon_prediction_features(
                estimator=estimator,
                predictor=predictor,
                features=features,
            )
            return extract_positive_class_scores(
                predictor.predict_proba(ag_features, model=model_name)
            )

        try:
            probabilities = estimator.predict_proba(features, model=model_name)
        except TypeError as exc:
            raise ValueError(
                "model_name is only supported for AutoGluon predictors or "
                "estimators that accept a 'model' keyword argument."
            ) from exc

        return extract_positive_class_scores(probabilities)

    return extract_positive_class_scores(estimator.predict_proba(features))


def _find_best_f1_threshold_from_score_support(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | None:
    """Search the observed score support exactly for the best F1 threshold."""
    unique_labels = np.unique(y_true)
    if np.any(pd.isna(unique_labels)) or not np.all(np.isin(unique_labels, [0, 1, False, True])):
        return None

    y_true_binary = np.asarray(y_true, dtype=int)
    order = np.argsort(y_score, kind="mergesort")[::-1]
    sorted_scores = y_score[order]
    sorted_true = y_true_binary[order]

    true_positives = np.cumsum(sorted_true)
    false_positives = np.cumsum(1 - sorted_true)
    total_positives = int(sorted_true.sum())
    unique_score_mask = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]

    candidate_thresholds = sorted_scores[unique_score_mask]
    tp_at_threshold = true_positives[unique_score_mask]
    fp_at_threshold = false_positives[unique_score_mask]
    fn_at_threshold = total_positives - tp_at_threshold
    denominator = (2 * tp_at_threshold) + fp_at_threshold + fn_at_threshold
    f1_scores = np.divide(
        2 * tp_at_threshold,
        denominator,
        out=np.zeros_like(candidate_thresholds, dtype=float),
        where=denominator > 0,
    )

    best_score = np.max(f1_scores)
    best_thresholds = candidate_thresholds[np.isclose(f1_scores, best_score)]

    return float(np.min(best_thresholds))


def _build_threshold_search_candidates(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_thresholds: int,
) -> np.ndarray:
    """Build threshold candidates from the score distribution and positive scores."""
    quantile_count = max(2, min(int(num_thresholds), y_score.shape[0]))
    distribution_thresholds = np.quantile(
        y_score,
        np.linspace(0.0, 1.0, quantile_count),
    )
    positive_score_thresholds = np.unique(y_score[np.asarray(y_true).reshape(-1) == 1])
    boundary_thresholds = np.array([0.0, float(np.min(y_score)), float(np.max(y_score)), 1.0])

    return np.unique(
        np.clip(
            np.concatenate(
                [distribution_thresholds, positive_score_thresholds, boundary_thresholds]
            ),
            0.0,
            1.0,
        )
    )


def find_best_threshold(
    y_true,
    y_score,
    num_thresholds: int = 200,
    thresholds=None,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> float:
    """
    Search decision thresholds directly on positive-class score arrays.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    y_score : array-like
        Positive-class scores.

    num_thresholds : int, default=200
        Number of quantile-based thresholds to evaluate from the score
        distribution when thresholds is not provided and a custom scorer is used.

    thresholds : array-like, optional
        Explicit candidate thresholds.

    scorer : callable, optional
        Function with signature scorer(y_true, y_pred). Defaults to F1-score.

    Returns
    -------
    float
        Threshold with the best score under the chosen scorer.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = extract_positive_class_scores(y_score).reshape(-1)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same number of observations.")

    if thresholds is None:
        if num_thresholds < 1:
            raise ValueError("num_thresholds must be at least 1.")
        if scorer is None:
            exact_f1_threshold = _find_best_f1_threshold_from_score_support(y_true, y_score)
            if exact_f1_threshold is not None:
                return exact_f1_threshold

        thresholds = _build_threshold_search_candidates(
            y_true=y_true,
            y_score=y_score,
            num_thresholds=num_thresholds,
        )
    else:
        thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
        if thresholds.size == 0:
            raise ValueError("thresholds must contain at least one value.")
        thresholds = np.unique(np.sort(thresholds))

    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("thresholds must be in the interval [0, 1].")

    if scorer is None:
        scorer = lambda y_true_input, y_pred_input: f1_score(
            y_true_input,
            y_pred_input,
            zero_division=0,
        )

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        current_score = float(scorer(y_true, y_pred))
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)

    return float(best_threshold)


def _coerce_results_frame(results) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, pd.Series):
        return pd.DataFrame([results.to_dict()])
    if isinstance(results, dict):
        return pd.DataFrame([results])
    return pd.DataFrame(results)


_RESULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("Model",),
    "Model": ("model",),
}


def _resolve_results_column_name(results_df: pd.DataFrame, column_name: str) -> str:
    candidate_names = (column_name,) + _RESULT_COLUMN_ALIASES.get(column_name, ())
    for candidate_name in candidate_names:
        if candidate_name in results_df.columns:
            return candidate_name

    raise KeyError(f"Column '{column_name}' not found in results.")


def _select_single_result_row(results, filters: dict | None = None) -> pd.Series:
    results_df = _coerce_results_frame(results)

    if filters:
        mask = pd.Series(True, index=results_df.index)
        for column, value in filters.items():
            resolved_column = _resolve_results_column_name(results_df, column)
            mask &= results_df[resolved_column].eq(value)
        results_df = results_df.loc[mask]

    if results_df.empty:
        raise ValueError("No rows matched the requested filters.")
    if len(results_df) > 1:
        raise ValueError("Multiple rows matched the requested filters. Add more specific filters.")

    return results_df.iloc[0]


_METRIC_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "f1": ("F1-score",),
    "F1-score": ("f1",),
}


def _resolve_metric_name(row: pd.Series, metric_name: str) -> str:
    candidate_names = (metric_name,) + _METRIC_NAME_ALIASES.get(metric_name, ())
    for candidate_name in candidate_names:
        if candidate_name in row.index:
            return candidate_name

    raise KeyError(f"Metric '{metric_name}' not found in results.")


def _extract_metric_value_from_row(row: pd.Series, metric_name: str) -> float:
    resolved_metric_name = _resolve_metric_name(row, metric_name)
    value = row[resolved_metric_name]
    if pd.isna(value):
        raise ValueError(f"Metric '{metric_name}' is missing.")

    return float(value)


def _compare_metric_values(
    left_value: float,
    right_value: float,
    better_when: str = "higher",
    tolerance: float = 1e-4,
) -> str:
    if better_when not in {"higher", "lower"}:
        raise ValueError("better_when must be either 'higher' or 'lower'.")

    if abs(left_value - right_value) <= tolerance:
        return "tie"

    if better_when == "higher":
        return "left" if left_value > right_value else "right"

    return "left" if left_value < right_value else "right"


def _format_metric_value(value: float, decimals: int = 4) -> str:
    return f"{float(value):.{decimals}f}"


def _build_result_label(
    strategy_name: str,
    filters: dict | None = None,
    label: str | None = None,
) -> str:
    if label is not None:
        return label

    filters = filters or {}
    detail_values = [str(value) for column, value in filters.items() if column != "Strategy"]
    if detail_values:
        return f"{strategy_name} ({', '.join(detail_values)})"

    return strategy_name


def _resolve_metric_direction(metric_name: str, better_when: dict[str, str] | None = None) -> str:
    if better_when and metric_name in better_when:
        direction = better_when[metric_name]
    elif metric_name.lower() in {"log_loss", "loss", "rmse", "mae", "mse"}:
        direction = "lower"
    elif metric_name.lower() in {
        "roc_auc",
        "score_test",
        "average_precision",
        "ap",
        "f1",
        "f1-score",
        "precision",
        "recall",
    }:
        direction = "higher"
    else:
        raise ValueError(
            f"No comparison direction was provided for metric '{metric_name}'. "
            "Pass better_when explicitly for this metric."
        )

    if direction not in {"higher", "lower"}:
        raise ValueError("better_when values must be either 'higher' or 'lower'.")

    return direction


def build_threshold_comparison_text(
    default_results=None,
    optimized_results=None,
    model_name: str | None = None,
    dataset_name: str = "test set",
    metric_name: str = "f1",
    metric_label: str = "F1-score",
    model_column: str = "model",
    default_threshold: float = 0.5,
    optimized_threshold: float | None = None,
    tolerance: float = 1e-4,
    default_metric: float | None = None,
    optimized_metric: float | None = None,
    model_label: str | None = None,
) -> str:
    """
    Build concise text comparing default and optimized thresholds.

    This helper supports either direct metric values or result tables that
    contain a single row for the selected model.
    """
    if (default_metric is None) != (optimized_metric is None):
        raise ValueError(
            "default_metric and optimized_metric must both be provided when using direct values."
        )

    if default_metric is None:
        if default_results is None or optimized_results is None:
            raise ValueError(
                "Provide either direct metrics or both default_results and optimized_results."
            )
        filters = {model_column: model_name} if model_name is not None else None
        default_row = _select_single_result_row(default_results, filters=filters)
        optimized_row = _select_single_result_row(optimized_results, filters=filters)
        default_metric = _extract_metric_value_from_row(default_row, metric_name)
        optimized_metric = _extract_metric_value_from_row(optimized_row, metric_name)

    default_metric = float(default_metric)
    optimized_metric = float(optimized_metric)
    comparison = _compare_metric_values(
        default_metric,
        optimized_metric,
        better_when="higher",
        tolerance=tolerance,
    )

    model_label = model_label or model_name or "the selected model"
    if comparison == "tie":
        verdict = f"threshold tuning does not materially change the {metric_label} for {model_label}"
    elif comparison == "right":
        verdict = f"threshold tuning improves the {metric_label} for {model_label}"
    else:
        verdict = f"the default threshold still gives the higher {metric_label} for {model_label}"

    default_sentence = (
        f"The default threshold of {default_threshold:.3f} gives {metric_label} "
        f"{_format_metric_value(default_metric)}."
    )
    if optimized_threshold is None:
        optimized_sentence = (
            f"The optimized threshold gives {metric_label} "
            f"{_format_metric_value(optimized_metric)}."
        )
    else:
        optimized_sentence = (
            f"The optimized threshold of {optimized_threshold:.3f} gives {metric_label} "
            f"{_format_metric_value(optimized_metric)}."
        )

    return f"On the {dataset_name}, {verdict}. {default_sentence} {optimized_sentence}"


def build_strategy_metric_comparison_text(
    results,
    left_strategy: str,
    right_strategy: str,
    metrics: list[str],
    left_filters: dict | None = None,
    right_filters: dict | None = None,
    left_label: str | None = None,
    right_label: str | None = None,
    metric_labels: dict[str, str] | None = None,
    better_when: dict[str, str] | None = None,
    context_name: str = "the current summary",
    tolerance: float = 1e-4,
) -> str:
    left_filters = {**(left_filters or {}), "Strategy": left_strategy}
    right_filters = {**(right_filters or {}), "Strategy": right_strategy}

    left_row = _select_single_result_row(results, filters=left_filters)
    right_row = _select_single_result_row(results, filters=right_filters)

    left_label = _build_result_label(left_strategy, filters=left_filters, label=left_label)
    right_label = _build_result_label(right_strategy, filters=right_filters, label=right_label)
    metric_labels = metric_labels or {}
    better_when = better_when or {}

    sentences = [f"In {context_name}, comparing {left_label} with {right_label}."]

    for metric_name in metrics:
        metric_label = metric_labels.get(metric_name, metric_name)
        left_value = _extract_metric_value_from_row(left_row, metric_name)
        right_value = _extract_metric_value_from_row(right_row, metric_name)
        metric_direction = _resolve_metric_direction(metric_name, better_when=better_when)
        comparison = _compare_metric_values(
            left_value,
            right_value,
            better_when=metric_direction,
            tolerance=tolerance,
        )

        if comparison == "tie":
            sentences.append(
                f"{metric_label} is similar ({left_label}: {_format_metric_value(left_value)}, "
                f"{right_label}: {_format_metric_value(right_value)})."
            )
        elif comparison == "left":
            sentences.append(
                f"{left_label} is better on {metric_label} "
                f"({_format_metric_value(left_value)} versus {_format_metric_value(right_value)})."
            )
        else:
            sentences.append(
                f"{right_label} is better on {metric_label} "
                f"({_format_metric_value(right_value)} versus {_format_metric_value(left_value)})."
            )

    return " ".join(sentences)


def build_running_metric_row(
    strategy_name: str,
    model_name: str,
    y_tuning,
    tuning_scores,
    y_test,
    test_scores,
    best_threshold: float | None = None,
    threshold_scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """
    Build one running-summary row from tuning-set scores and test-set scores.
    """
    tuning_scores = extract_positive_class_scores(tuning_scores).reshape(-1)
    test_scores = extract_positive_class_scores(test_scores).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    if y_test.shape[0] != test_scores.shape[0]:
        raise ValueError("y_test and test_scores must have the same number of observations.")

    if best_threshold is None:
        best_threshold = find_best_threshold(
            y_true=y_tuning,
            y_score=tuning_scores,
            scorer=threshold_scorer,
        )

    y_pred_test = (test_scores >= best_threshold).astype(int)

    return {
        "Strategy": strategy_name,
        "Model": model_name,
        "Test ROC-AUC": roc_auc_score(y_test, test_scores),
        "Test Average Precision": average_precision_score(y_test, test_scores),
        "best_threshold": float(best_threshold),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "F1-score": f1_score(y_test, y_pred_test, zero_division=0),
    }


def append_running_metric_summary(
    rows: list[dict[str, Any]],
    running_metric_results: list[dict[str, Any]] | None = None,
    display_fn=None,
    round_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Update and display a running summary table keyed by strategy and model.
    """
    if running_metric_results is None:
        running_metric_results = []

    existing_rows = {
        (result["Strategy"], result["Model"]): idx
        for idx, result in enumerate(running_metric_results)
    }

    for row in rows:
        row_key = (row["Strategy"], row["Model"])
        if row_key in existing_rows:
            running_metric_results[existing_rows[row_key]] = row
        else:
            running_metric_results.append(row)
            existing_rows[row_key] = len(running_metric_results) - 1

    running_metric_df = pd.DataFrame(running_metric_results)
    round_map = round_map or {
        "Test ROC-AUC": 4,
        "Test Average Precision": 4,
        "best_threshold": 4,
        "precision": 4,
        "recall": 4,
        "F1-score": 4,
    }
    rounded_df = running_metric_df.round({
        column: decimals
        for column, decimals in round_map.items()
        if column in running_metric_df.columns
    })

    if display_fn is None:
        try:
            from IPython.display import display as ipython_display
            display_fn = ipython_display
        except ImportError:
            display_fn = None

    if display_fn is not None:
        display_fn(rounded_df)

    return rounded_df


def plot_named_calibration_curves(
    y_true,
    named_scores,
    n_bins: int = 20,
    ax=None,
    title: str = "Calibration Plot",
    curve_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Plot multiple named calibration curves from score arrays.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    named_scores : dict or sequence of tuples
        Mapping from display name to score array.

    n_bins : int, default=20
        Number of bins for each calibration curve.

    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    title : str, default="Calibration Plot"
        Plot title.

    curve_kwargs : dict, optional
        Per-curve keyword arguments passed to CalibrationDisplay.from_predictions.

    Returns
    -------
    dict
        Dictionary containing the figure, axes, and calibration displays.
    """
    if isinstance(named_scores, dict):
        score_items = list(named_scores.items())
    else:
        score_items = list(named_scores)

    if not score_items:
        raise ValueError("named_scores must contain at least one score array.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    y_true = np.asarray(y_true).reshape(-1)
    curve_kwargs = curve_kwargs or {}
    displays = {}

    for index, (name, scores) in enumerate(score_items):
        display_kwargs = dict(curve_kwargs.get(name, {}))
        display_kwargs.setdefault("ref_line", index == 0)
        displays[name] = CalibrationDisplay.from_predictions(
            y_true=y_true,
            y_prob=extract_positive_class_scores(scores),
            n_bins=n_bins,
            name=name,
            ax=ax,
            **display_kwargs,
        )

    ax.set_title(title)

    return {
        "figure": fig,
        "ax": ax,
        "displays": displays,
    }


# =============================================================================
# AUTOGLUON SKLEARN WRAPPER
# =============================================================================

class AutoGluonSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for AutoGluon TabularPredictor.
    
    Inherits from scikit-learn's BaseEstimator and ClassifierMixin to provide
    full compatibility with estimator-aware scikit-learn tools such as
    PartialDependenceDisplay().
    
    Parameters
    ----------
    label : str
        Name of the target column.
    
    predictor_args : dict, optional
        Additional arguments passed to TabularPredictor()
        (e.g., problem_type, eval_metric, path, sample_weight).
        Default is None (empty dict).
    
    fit_args : dict, optional
        Additional arguments passed to TabularPredictor.fit() method
        (e.g., holdout_frac, presets, time_limit, excluded_model_types).
        Default is None (empty dict).
    
    n_jobs : int, optional
        Number of CPU cores to use. If None, uses AutoGluon defaults.
    
    Attributes
    ----------
    predictor_ : TabularPredictor
        The trained AutoGluon predictor.
    
    classes_ : ndarray
        Class labels (for classification tasks).
    
    n_features_in_ : int
        Number of features seen during fit.
    
    feature_names_ : list
        Feature names inferred during fitting.
    
    is_fitted_ : bool
        Whether the estimator has been fitted.
    
    predictor_path_ : str
        Absolute path to the saved predictor.
    
    Examples
    --------
    >>> wrapper = AutoGluonSklearnWrapper(
    ...     label='target',
    ...     predictor_args={'problem_type': 'binary', 'eval_metric': 'roc_auc'},
    ...     fit_args={'presets': 'medium_quality', 'time_limit': 300}
    ... )
    >>> wrapper.fit(X_train, y_train)
    >>> predictions = wrapper.predict(X_test)
    """
    
    _estimator_type = "classifier"

    def _more_tags(self):
        return {"estimator_type": "classifier"}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
    
    def __init__(self, label: str, predictor_args: Optional[Dict] = None, 
                 fit_args: Optional[Dict] = None, n_jobs: Optional[int] = None):
        self.label = label
        self.predictor_args = predictor_args if predictor_args else {}
        self.fit_args = fit_args if fit_args else {}
        self.n_jobs = n_jobs

    def _validate_features(self, X) -> pd.DataFrame:
        """
        Validate feature names/counts and return a DataFrame with training features.
        
        Parameters
        ----------
        X : array-like, DataFrame, or polars.DataFrame
            Input data to validate.
        
        Returns
        -------
        pd.DataFrame
            Validated DataFrame with correct column names.
        
        Raises
        ------
        ValueError
            If feature names or counts don't match training data.
        """
        # Handle Polars DataFrame
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X = X.to_pandas()
        except ImportError:
            pass
        
        if hasattr(X, "columns"):
            cols = list(X.columns)
            if cols != self.feature_names_:
                raise ValueError("Feature names do not match training data.")
            return X.copy()

        if X.shape[1] != len(self.feature_names_):
            raise ValueError("Number of features does not match training data.")

        return pd.DataFrame(X, columns=self.feature_names_)

    def __sklearn_is_fitted__(self) -> bool:
        """Official scikit-learn API for checking fitted status."""
        return getattr(self, "is_fitted_", False)

    def __getstate__(self) -> Dict:
        """Make the wrapper pickle-friendly for joblib/Parallel."""
        state = self.__dict__.copy()
        predictor_path = state.get("predictor_path_", None)
        predictor = state.get("predictor_", None)
        if predictor_path is None and predictor is not None:
            predictor_path = predictor.path
        if predictor_path is not None:
            state["predictor_path_"] = os.path.abspath(predictor_path)
        state["predictor_"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        """Restore predictor on unpickle."""
        self.__dict__.update(state)
        predictor_path = self.__dict__.get("predictor_path_", None)
        if predictor_path:
            self.predictor_ = TabularPredictor.load(predictor_path)
        self.is_fitted_ = predictor_path is not None

    @property
    def predictor(self):
        """Backward-compatible access to the fitted AutoGluon predictor."""
        return getattr(self, "predictor_", None)

    def fit(self, X, y, sample_weight=None):
        """
        Fit AutoGluon model using scikit-learn interface.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights to pass to AutoGluon if configured.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, dtype=None)
        y = y_checked

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
            train_df = X.copy()
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X_checked.shape[1])]
            train_df = pd.DataFrame(X_checked, columns=self.feature_names_)

        self.n_features_in_ = train_df.shape[1]

        predictor_args = dict(self.predictor_args)
        if sample_weight is not None:
            weight_col = predictor_args.get("sample_weight")
            if weight_col is None:
                weight_col = "sample_weight"
                predictor_args["sample_weight"] = weight_col
            train_df[weight_col] = sample_weight

        train_df[self.label] = y
        train_data = TabularDataset(train_df)

        fit_args = dict(self.fit_args)
        if self.n_jobs is not None and fit_args.get("num_cpus") is None:
            fit_args["num_cpus"] = self.n_jobs
        
        # Remove sample_weight from fit_args if present (TabularPredictor.fit doesn't accept it)
        fit_args_clean = {k: v for k, v in fit_args.items() if k != "sample_weight"}

        self.predictor_ = TabularPredictor(
            label=self.label,
            **predictor_args
        ).fit(train_data, **fit_args_clean)

        self.predictor_path_ = os.path.abspath(self.predictor_.path)

        if self.predictor_.problem_type in ['binary', 'multiclass']:
            self.classes_ = np.array(self.predictor_.class_labels)

        self.is_fitted_ = True

        return self 

    def predict(self, X):
        """
        Make class predictions.
        
        Parameters
        ----------
        X : {array-like, sparse matrix, polars.DataFrame} of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        
        # Handle Polars DataFrame
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X = X.to_pandas()
        except ImportError:
            pass
        
        if hasattr(X, "columns"):
            check_array(X, accept_sparse=False, dtype=None)
            df = self._validate_features(X)
        else:
            X_checked = check_array(X, accept_sparse=False, dtype=None)
            df = self._validate_features(X_checked)

        return self.predictor_.predict(TabularDataset(df)).values

    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : {array-like, sparse matrix, polars.DataFrame} of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        
        # Handle Polars DataFrame
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X = X.to_pandas()
        except ImportError:
            pass
        
        if hasattr(X, "columns"):
            check_array(X, accept_sparse=False, dtype=None)
            df = self._validate_features(X)
        else:
            X_checked = check_array(X, accept_sparse=False, dtype=None)
            df = self._validate_features(X_checked)

        return self.predictor_.predict_proba(TabularDataset(df)).values

    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'label': self.label,
            'predictor_args': self.predictor_args,
            'fit_args': self.fit_args,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params) -> 'AutoGluonSklearnWrapper':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if param == 'label':
                self.label = value
            elif param == 'predictor_args':
                self.predictor_args = value
            elif param == 'fit_args':
                self.fit_args = value
            elif param == 'n_jobs':
                self.n_jobs = value
            else:
                setattr(self, param, value)
        return self

    def _check_n_features(self, X, reset: bool = False) -> None:
        """Validate number of features."""
        n_features = X.shape[1]
        if reset:
            self.n_features_in_ = n_features
        elif n_features != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {n_features}")

    def _check_feature_names(self, X, reset: bool = False) -> None:
        """Validate feature names (AutoGluon requirement)."""
        if reset:
            if isinstance(X, np.ndarray):
                self.feature_names_ = [f'feat_{i}' for i in range(X.shape[1])]
            else:
                self.feature_names_ = X.columns.tolist()
        elif hasattr(X, 'columns'):
            if list(X.columns) != self.feature_names_:
                raise ValueError("Feature names mismatch between fit and predict")


def load_autogluon(folder_path: str, persist_model: bool = False) -> AutoGluonSklearnWrapper:
    """
    Load a pre-trained AutoGluon TabularPredictor into an AutoGluonSklearnWrapper.
    
    Parameters
    ----------
    folder_path : str
        Path to the directory containing the saved AutoGluon predictor.
    
    persist_model : bool, default=False
        If True, calls predictor.persist() after loading to keep models in memory
        for faster predictions.
    
    Returns
    -------
    AutoGluonSklearnWrapper
        Wrapper instance containing the loaded predictor and metadata.
    
    Raises
    ------
    FileNotFoundError
        If the specified folder_path does not exist.
    Exception
        If any other error occurs during loading.
    """
    try:
        print(f"Loading AutoGluon predictor from: {folder_path}")
        predictor = TabularPredictor.load(folder_path)
        print("Predictor loaded successfully.")

        # Extract metadata
        label = predictor.label
        feature_names = predictor.feature_metadata_in.get_features()
        n_features = len(feature_names)
        classes = None
        if predictor.problem_type in ['binary', 'multiclass']:
            classes = np.array(predictor.class_labels)

        # Create wrapper instance
        wrapper = AutoGluonSklearnWrapper(label=label, predictor_args={}, fit_args={})
        wrapper.predictor_ = predictor
        wrapper.classes_ = classes
        wrapper.n_features_in_ = n_features
        wrapper.feature_names_ = feature_names
        wrapper.is_fitted_ = True

        if persist_model:
            print("Persisting predictor models in memory...")
            wrapper.predictor_.persist()
            print("Predictor models persisted.")

        print("AutoGluonSklearnWrapper created successfully.")
        return wrapper

    except FileNotFoundError:
        print(f"Error: Predictor directory not found at {folder_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the predictor: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def show_pdp(wrappedAGModel: BaseEstimator,
             list_features: List[str],
             list_categ_features: List[str],
             df: pd.DataFrame,
             xGTzero: bool = False,
             sampSize: int = 40000,
             show_ice: bool = False,
             n_jobs: int = 2) -> None:
    """
    Generate Partial Dependence Plots (PDP) and optionally ICE plots.
    
    Parameters
    ----------
    wrappedAGModel : BaseEstimator
        A fitted sklearn-compatible model wrapper.
    
    list_features : list of str
        List of feature names to plot.
    
    list_categ_features : list of str
        List of categorical feature names.
    
    df : pd.DataFrame
        DataFrame containing the features.
    
    xGTzero : bool, default=False
        If True, set x-axis lower limit to 0.
    
    sampSize : int, default=40000
        Number of samples to use for PDP calculation.
    
    show_ice : bool, default=False
        If True, show ICE lines in addition to PDP.
    
    n_jobs : int, default=2
        Number of parallel jobs to run.
    """
    for feature in list_features:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)

        plt.rcParams.update({'font.size': 16})

        plot_kind = 'both' if show_ice else 'average'
        ice_subsample = 250 if show_ice else None
        ice_lines_kw = {"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5} if show_ice else None
        pd_line_kw = {"color": "tab:red", "linestyle": "--", "linewidth": 2}

        # Get expected feature order (if available)
        if hasattr(wrappedAGModel, 'feature_names_'):
            feature_order = wrappedAGModel.feature_names_
            X_sample = df[feature_order].sample(min(sampSize, len(df)), random_state=2025)
        else:
            X_sample = df.sample(min(sampSize, len(df)), random_state=2025)

        disp = PartialDependenceDisplay.from_estimator(
            estimator=wrappedAGModel,
            X=X_sample,
            features=[feature],
            categorical_features=list_categ_features if list_categ_features else None,
            method='brute',
            kind=plot_kind,
            subsample=ice_subsample,
            ice_lines_kw=ice_lines_kw,
            pd_line_kw=pd_line_kw,
            percentiles=(0.0001, 0.9999),
            grid_resolution=100,
            ax=ax,
            random_state=2025,
            n_jobs=n_jobs
        )

        plot_title = f"Partial Dependence for {feature}"
        if show_ice:
            plot_title += " (with ICE)"
        ax.set_title(plot_title)

        for a in fig.get_axes():
            coordy = a.get_ylim()
            y_bottom = 0 if not show_ice else min(0, coordy[0])
            a.set_ylim(bottom=y_bottom, top=coordy[1] * 1.1)

        if xGTzero:
            for a in fig.get_axes():
                if feature in df.columns:
                    max_val = np.percentile(df[feature].dropna().values, 99.99)
                    a.set_xlim(left=0, right=max_val)
                else:
                    max_val = a.get_xlim()[1]
                    a.set_xlim(left=0, right=max_val)

        plt.show()
        plt.close('all')


def show_ale(wrappedAGModel: BaseEstimator,
             list_features: List[str],
             df: pd.DataFrame,
             xGTzero: bool = False,
             sampSize: int = 40000) -> None:
    """
    Generate Accumulated Local Effects (ALE) plots.
    
    Parameters
    ----------
    wrappedAGModel : BaseEstimator
        A fitted sklearn-compatible model wrapper.
    
    list_features : list of str
        List of feature names to plot.
    
    df : pd.DataFrame
        DataFrame containing the features.
    
    xGTzero : bool, default=False
        If True, set x-axis lower limit to 0.
    
    sampSize : int, default=40000
        Number of samples to use for ALE calculation.
    """
    try:
        from PyALE import ale
    except ImportError:
        print("PyALE not installed. Install with: pip install PyALE")
        return

    def do_nothing(data):
        return data

    # Get predictor features (handle both Pipeline and direct wrapper)
    if hasattr(wrappedAGModel, 'named_steps') and 'autogluon' in wrappedAGModel.named_steps:
        predictor_features = wrappedAGModel.named_steps['autogluon'].predictor.features()
    elif hasattr(wrappedAGModel, 'feature_names_'):
        predictor_features = wrappedAGModel.feature_names_
    else:
        predictor_features = None

    for feature in list_features:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)

        plt.rcParams.update({'font.size': 12})

        ale_kwargs = dict(
            X=df.sample(sampSize, random_state=2025),
            model=wrappedAGModel,
            feature=[feature],
            include_CI=False,
            C=0.9999,
            grid_size=100,
            encode_fun=do_nothing,
            fig=fig,
            ax=ax
        )
        if predictor_features is not None:
            ale_kwargs['predictors'] = predictor_features

        ale_eff = ale(**ale_kwargs)
        
        if xGTzero:
            for a in fig.get_axes():
                max_val = np.percentile(df[feature].values, 99.99)
                a.set_xlim(left=0, right=max_val)

        plt.show()
        plt.close('all')


def show_calibration_plot(estimator, X, y, name: str = 'Model', 
                          n_bins: int = 20, color: str = 'orange',
                          ax=None, title: str = "Calibration Plot"):
    """
    Generate a calibration plot for a classifier.
    
    Parameters
    ----------
    estimator : object
        Fitted classifier with predict_proba method.
    
    X : array-like
        Feature data.
    
    y : array-like
        True labels.
    
    name : str, default='Model'
        Name for the legend.
    
    n_bins : int, default=20
        Number of bins for calibration.
    
    color : str, default='orange'
        Color for the calibration curve.
    
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    title : str, default="Calibration Plot"
        Title for the plot.
    """
    disp = CalibrationDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        n_bins=n_bins,
        name=name,
        color=color,
        ax=ax
    )
    
    if ax is None:
        disp.ax_.set_title(title)
    
    return disp


# =============================================================================
# CREDIT SCORING UTILITIES
# =============================================================================

def calculate_score(prob_default: np.ndarray, pdo: int = 40, base_score: int = 600) -> np.ndarray:
    """
    Convert probability of default to a 3-digit credit score.
    
    Uses the standard credit scoring formula:
    Score = BaseScore - Factor * log(OddsBad)
    where OddsBad = PD / (1 - PD) and Factor = PDO / log(2)
    
    Parameters
    ----------
    prob_default : array-like
        Predicted probability of default.
    
    pdo : int, default=40
        Points to Double the Odds. The number of points 
        required for the odds to double.
    
    base_score : int, default=600
        Base score corresponding to odds of 1:1.
    
    Returns
    -------
    np.ndarray
        Credit scores clipped to range [300, 850].
    """
    eps = 1e-9
    prob_default = np.clip(prob_default, eps, 1 - eps)
    odds_bad = prob_default / (1 - prob_default)
    factor = pdo / np.log(2)
    score = base_score - factor * np.log(odds_bad)
    return np.clip(score, 300, 850).astype(int)


def score_to_probability(score: np.ndarray, pdo: int = 40, base_score: int = 600) -> np.ndarray:
    """
    Convert credit score back to probability of default.
    
    Inverse of calculate_score function.
    
    Parameters
    ----------
    score : array-like
        Credit scores.
    
    pdo : int, default=40
        Points to Double the Odds.
    
    base_score : int, default=600
        Base score corresponding to odds of 1:1.
    
    Returns
    -------
    np.ndarray
        Probability of default [0, 1].
    """
    factor = pdo / np.log(2)
    odds_bad = np.exp((score - base_score) / -factor)
    prob_default = odds_bad / (1 + odds_bad)
    return prob_default


# =============================================================================
# KS TABLE AND EVALUATION UTILITIES
# =============================================================================

def ks_table(data: pd.DataFrame, y_true_col: str, y_pred_col: str, 
             n_bins: int = 10, is_score: bool = False, 
             sample_weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a KS (Kolmogorov-Smirnov) table for model evaluation.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing true labels and predicted probabilities/scores.
    
    y_true_col : str
        Name of column with true binary labels (0 or 1).
    
    y_pred_col : str
        Name of column with predicted probabilities (higher=riskier) 
        or scores (lower=riskier).
    
    n_bins : int, default=10
        Number of bins to divide values into.
    
    is_score : bool, default=False
        True if y_pred_col contains scores (lower=riskier),
        False if it contains probabilities (higher=riskier).
    
    sample_weight_col : str, optional
        Name of column containing sample weights.
    
    Returns
    -------
    pd.DataFrame
        KS table with cumulative bad/good percentages and KS statistic.
    """
    import duckdb
    
    # Select relevant columns
    cols_to_select = [y_true_col, y_pred_col]
    if sample_weight_col:
        cols_to_select.append(sample_weight_col)
    
    df = data[cols_to_select].copy()
    
    # Rename for internal use
    rename_map = {y_true_col: 'y_true', y_pred_col: 'y_pred'}
    if sample_weight_col in df.columns:
        rename_map[sample_weight_col] = 'weight'
        df['weight'] = pd.to_numeric(df[sample_weight_col], errors='coerce').fillna(1.0)
    else:
        df['weight'] = 1.0
    
    df.rename(columns=rename_map, inplace=True)
    df.dropna(subset=['y_pred'], inplace=True)
    
    if df.empty:
        print("Warning: No valid data points for KS table.")
        return pd.DataFrame()
    
    # Create bins
    try:
        df["bin"] = pd.qcut(df["y_pred"].rank(method='first'), q=n_bins, 
                           labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: pd.qcut failed ({e}). Using manual binning.")
        ranks = df["y_pred"].rank(method='first')
        bin_size = max(1, len(df) // n_bins)
        df['bin'] = ((ranks - 1) // bin_size).clip(upper=n_bins - 1).astype(int)
    
    # Determine sort order
    sort_ascending = is_score
    
    # Aggregate using duckdb
    ks_df = duckdb.query("""
        SELECT
            bin,
            MIN(y_pred) AS min_value,
            MAX(y_pred) AS max_value,
            AVG(y_pred) AS avg_value,
            SUM(weight) AS count,
            SUM(y_true * weight) AS bads
        FROM df
        GROUP BY bin
    """).to_df()
    
    ks_df = ks_df.reset_index().sort_values("min_value", ascending=sort_ascending)
    
    ks_df["goods"] = ks_df["count"] - ks_df["bads"]
    ks_df["bad_rate"] = np.where(ks_df["count"] > 1e-9, 
                                  ks_df["bads"] / ks_df["count"], 0)
    
    total_bads = ks_df["bads"].sum()
    total_goods = ks_df["goods"].sum()
    
    if total_bads <= 1e-9 or total_goods <= 1e-9:
        ks_df["cum_bads_pct"] = 0.0
        ks_df["cum_goods_pct"] = 0.0
        ks_df["ks"] = 0.0
        print("Warning: KS calculation skipped - total goods or bads is zero.")
    else:
        ks_df["cum_bads_pct"] = (ks_df["bads"].cumsum() / total_bads) * 100
        ks_df["cum_goods_pct"] = (ks_df["goods"].cumsum() / total_goods) * 100
        ks_df["ks"] = np.abs(ks_df["cum_bads_pct"] - ks_df["cum_goods_pct"])
    
    # Rename columns
    ks_df.rename(columns={
        'min_value': f'min_{y_pred_col}',
        'max_value': f'max_{y_pred_col}',
        'avg_value': f'avg_{y_pred_col}'
    }, inplace=True)
    
    # Print max KS info
    if 'ks' in ks_df.columns and not ks_df['ks'].empty and total_bads > 1e-9 and total_goods > 1e-9:
        max_ks = ks_df['ks'].max()
        max_ks_row = ks_df.loc[ks_df['ks'].idxmax()]
        print(f"KS Statistic (Max KS): {max_ks:.4f}")
        print(f"  Occurs in bin with range: [{max_ks_row[f'min_{y_pred_col}']:.4f} - {max_ks_row[f'max_{y_pred_col}']:.4f}]")
    
    # Select final columns
    final_cols = [
        f"min_{y_pred_col}", f"max_{y_pred_col}", f"avg_{y_pred_col}",
        "count", "bads", "goods", "bad_rate",
        "cum_bads_pct", "cum_goods_pct", "ks"
    ]
    final_cols = [col for col in final_cols if col in ks_df.columns]
    
    return ks_df[final_cols].reset_index(drop=True)


# =============================================================================
# BINNED PROBABILITY PLOT
# =============================================================================

def binned_prob_plot(
    data: pd.DataFrame,
    feature: str,
    target_binary: str,
    cont_feat_flag: Optional[bool] = None,
    transform_log_odds: bool = False,
    num_bins: int = 10,
    show_plot: bool = True
) -> Dict[str, Any]:
    """
    Plot average binary target against feature bins or categories.
    
    For continuous features, calculates Spearman correlation.
    For categorical features, calculates Mutual Information.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    
    feature : str
        Name of the feature to analyze.
    
    target_binary : str
        Name of the binary target variable.
    
    cont_feat_flag : bool, optional
        True if feature is continuous, False if categorical.
        If None, will be inferred from the data.
    
    transform_log_odds : bool, default=False
        If True, transform probabilities to log odds.
    
    num_bins : int, default=10
        Number of bins for continuous features.
    
    show_plot : bool, default=True
        If True, display the plot.
    
    Returns
    -------
    dict
        Dictionary with feature name, measure name, value, and p-value.
    """
    df = data.copy()
    
    # Infer feature type if not provided
    if cont_feat_flag is None:
        tmp = df[feature].dropna()
        sample_size = min(100, len(tmp))
        if sample_size > 0:
            tmp = tmp.sample(sample_size, replace=False, random_state=2025)
            cont_feat_flag = tmp.nunique() > min(60, sample_size * 0.5)
        else:
            cont_feat_flag = False
        print(f"Feature {feature} inferred as {'continuous' if cont_feat_flag else 'categorical'}.")
    
    # Create bins
    if cont_feat_flag:
        try:
            df["bin_label"] = pd.qcut(
                df[feature].rank(method='first'),
                q=num_bins,
                duplicates="drop",
                labels=[str(i) for i in range(1, num_bins + 1)]
            )
        except ValueError as e:
            print(f"Warning: pd.qcut failed for {feature} ({e}). Using fallback.")
            ranks = df[feature].rank(method='first')
            bin_size = max(1, len(df) // num_bins)
            df['bin_label'] = ((ranks - 1) // bin_size).clip(upper=num_bins - 1).astype(str)
    else:
        df["bin_label"] = df[feature].astype("category")
        df[feature + "_codes"] = df[feature].astype("category").cat.codes
    
    # Group and aggregate
    grouped = df.groupby("bin_label", observed=False).agg(
        **{
            "average_" + target_binary: (target_binary, "mean"),
            "count": (target_binary, "count")
        }
    )
    
    # Log-odds transform
    if transform_log_odds:
        eps = 1e-6
        grouped["transform_avg_prob"] = special.logit(
            np.clip(grouped["average_" + target_binary], eps, 1 - eps)
        )
    
    # Calculate statistics
    measure_name = None
    measure_value = None
    p_value = None
    
    if cont_feat_flag:
        measure_name = "spearman_corr"
        y = "transform_avg_prob" if transform_log_odds else "average_" + target_binary
        grouped_idx_numeric = pd.to_numeric(grouped.index, errors='coerce').fillna(0)
        if len(grouped) > 1:
            measure_value, p_value = stats.spearmanr(grouped_idx_numeric, grouped[y])
        else:
            measure_value, p_value = np.nan, np.nan
    else:
        measure_name = "mutual_info"
        df_mi = df[[feature + "_codes", target_binary]].dropna()
        if not df_mi.empty:
            measure_value = mutual_info_classif(
                df_mi[[feature + "_codes"]], df_mi[target_binary], 
                discrete_features=True
            )[0]
        else:
            measure_value = np.nan
        p_value = None
    
    # Plotting
    if show_plot:
        y_col = "transform_avg_prob" if transform_log_odds else "average_" + target_binary
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_x = range(len(grouped)) if cont_feat_flag else grouped.index
        
        ax.plot(
            plot_x, grouped[y_col],
            marker="o", linestyle="-",
            label="Log Odds" if transform_log_odds else "Probability"
        )
        ax.set_xlabel(feature, fontsize=14)
        ax.set_ylabel("Log Odds" if transform_log_odds else "Probability", fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        
        ax2 = ax.twinx()
        ax2.bar(plot_x, grouped["count"], alpha=0.25, color="gray", 
                align="center", label="Counts")
        ax2.set_ylabel("Counts", fontsize=14)
        ax2.tick_params(axis="y", labelsize=14)
        ax2.set_ylim(0, ax2.get_ylim()[1] * 10)
        
        ax.set_xticks(plot_x)
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=14)
        
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=14)
        
        title_suffix = (f" (Spearman: {measure_value:.3f})" if cont_feat_flag and measure_value is not None 
                        else f" (MI: {measure_value:.3f})" if not cont_feat_flag and measure_value is not None 
                        else "")
        ax.set_title(f"Binned Probability Plot for {feature}{title_suffix}", fontsize=16)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return {
        "feature": feature,
        "measure_name": measure_name,
        "measure_value": measure_value,
        "p_value": p_value,
        "log_odds": transform_log_odds
    }


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def parse_emp_length(x) -> int:
    """
    Convert employment length string to numeric years.
    
    Parameters
    ----------
    x : str or any
        Employment length string (e.g., "10+ years", "< 1 year", "n/a").
    
    Returns
    -------
    int
        Numeric representation of employment length:
        - 0 for "n/a" or "< 1 year"
        - 10 for "10+ years"
        - Extracted number for other formats
        - -1 for unexpected formats
    """
    if pd.isna(x) or x == "n/a":
        return 0
    elif "< 1 year" in str(x):
        return 0
    elif "10+ years" in str(x):
        return 10
    try:
        return int(re.findall(r'\d+', str(x))[0])
    except:
        return -1


def generate_eda_report(df: pd.DataFrame, title: str, output_path: str,
                        sample_frac: float = 0.1, random_state: int = 2025) -> None:
    """
    Generate and save a ydata-profiling report for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to profile.
    
    title : str
        Title for the report.
    
    output_path : str
        Path to save the HTML report.
    
    sample_frac : float, default=0.1
        Fraction of data to sample for large datasets.
    
    random_state : int, default=2025
        Random seed for sampling.
    """
    try:
        from ydata_profiling import ProfileReport
        
        print(f"Generating EDA report: {title}...")
        profile = ProfileReport(
            df.sample(frac=sample_frac, random_state=random_state) if sample_frac < 1.0 else df,
            title=title,
            progress_bar=False,
            duplicates=None,
            interactions=None
        )
        profile.to_file(output_path)
        print(f"Report saved to: {output_path}")
    except Exception as e:
        print(f"Error generating report '{title}': {e}")


# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def split_data(df: pd.DataFrame, target_col: str = None, train_size: float = 0.6, 
               calib_size_rel: float = 0.5, random_state: int = 2025) -> tuple:
    """
    Split data into train, calibration, and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    
    target_col : str, optional
        Name of target column for stratification. If None, no stratification.
    
    train_size : float, default=0.6
        Proportion of data for training set.
    
    calib_size_rel : float, default=0.5
        Proportion of remaining data for calibration (rest goes to test).
    
    random_state : int, default=2025
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (df_train, df_calib, df_test) DataFrames.
    """
    from sklearn.model_selection import train_test_split
    
    print(f"Splitting data (stratify={'Yes' if target_col else 'No'})...")
    y = df[target_col] if target_col else None

    if target_col:
        stratify_param = y
        # First split: Train and Temp (with stratification)
        df_train, df_temp, y_train, y_temp = train_test_split(
            df, y, train_size=train_size, random_state=random_state, stratify=stratify_param
        )
        # Second split: Temp into Calibration and Test (with stratification)
        stratify_param_temp = y_temp
        df_calib, df_test, y_calib, y_test = train_test_split(
            df_temp, y_temp, test_size=calib_size_rel, random_state=random_state, stratify=stratify_param_temp
        )
    else:
        # First split: Train and Temp (without stratification)
        df_train, df_temp = train_test_split(
            df, train_size=train_size, random_state=random_state, stratify=None
        )
        # Second split: Temp into Calibration and Test (without stratification)
        df_calib, df_test = train_test_split(
            df_temp, test_size=calib_size_rel, random_state=random_state, stratify=None
        )
        # Assign None to y splits as they don't exist
        y_train, y_temp, y_calib, y_test = None, None, None, None

    print(f"  Train shape: {df_train.shape}")
    print(f"  Calibration shape: {df_calib.shape}")
    print(f"  Test shape: {df_test.shape}")
    if target_col:
        print(f"  Train target mean: {y_train.mean():.4f}")
        print(f"  Calibration target mean: {y_calib.mean():.4f}")
        print(f"  Test target mean: {y_test.mean():.4f}")

    # Return copies
    return df_train.copy(), df_calib.copy(), df_test.copy()


def train_autogluon_model(
    df_train: pd.DataFrame,
    label: str,
    weight_col: str,
    modeling_features: list,
    model_folder: str,
    predictor_args: dict,
    fit_args: dict
) -> 'AutoGluonSklearnWrapper':
    """
    Train an AutoGluon model using the Sklearn wrapper.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    
    label : str
        Name of target column.
    
    weight_col : str or None
        Name of sample weight column.
    
    modeling_features : list
        List of feature names to use.
    
    model_folder : str
        Path to save the model.
    
    predictor_args : dict
        Arguments for TabularPredictor.
    
    fit_args : dict
        Arguments for TabularPredictor.fit().
    
    Returns
    -------
    AutoGluonSklearnWrapper
        Trained model wrapper.
    """
    print(f"--- Training AutoGluon Model in: {model_folder} ---")
    remove_ag_folder(model_folder)  # Clean up previous runs

    print(f"Using features: {modeling_features}")
    print(f"Training data shape: {df_train.shape}")
    if weight_col:
        print(f"Sum of weights in training data: {df_train[weight_col].sum():.2f}")

    start_time = time.time()

    # Prepare X, y, and weights for the wrapper's fit method
    X_train_ag = df_train[modeling_features]
    y_train_ag = df_train[label]
    weights_train_ag = df_train[weight_col] if weight_col else None

    # Update predictor_args with sample_weight if provided
    if weight_col:
        predictor_args['sample_weight'] = weight_col

    ag_model_wrapped = AutoGluonSklearnWrapper(
        label=label,
        predictor_args=predictor_args,
        fit_args=fit_args
    )

    # Fit the model
    ag_model_wrapped.fit(X_train_ag, y_train_ag, sample_weight=weights_train_ag)

    end_time = time.time()
    print(f"AutoGluon training completed in {end_time - start_time:.2f} seconds.")

    # Display leaderboard
    ag_predictor = ag_model_wrapped.predictor
    if ag_predictor:
        print("\nAutoGluon Leaderboard:")
        leaderboard = ag_predictor.leaderboard(silent=True)
        print(leaderboard)
    else:
        print("Could not access underlying predictor to display leaderboard.")

    return ag_model_wrapped


def summarize_ttd_by_source(df_ttd: pd.DataFrame, target_col: str = 'default_flag', 
                            weight_col: str = 'sample_weight', 
                            source_col: str = 'source') -> pd.DataFrame:
    """
    Calculate summary statistics for a TTD DataFrame grouped by source.
    
    Parameters
    ----------
    df_ttd : pd.DataFrame
        TTD DataFrame with source column.
    
    target_col : str, default='default_flag'
        Name of target column.
    
    weight_col : str, default='sample_weight'
        Name of weight column.
    
    source_col : str, default='source'
        Name of source column to group by.
    
    Returns
    -------
    pd.DataFrame
        Summary statistics by source.
    """
    print(f"\n--- Summarizing TTD Data by '{source_col}' ---")
    # Step 1: Calculate standard aggregations
    summary_stats = df_ttd.groupby(source_col).agg(
        row_count=(target_col, 'size'),
        sum_weights=(weight_col, 'sum'),
        unweighted_default_rate=(target_col, 'mean')
    ).reset_index()

    # Step 2: Calculate weighted default rate separately using apply
    weighted_rates = df_ttd.groupby(source_col).apply(
        lambda x: np.average(x[target_col], weights=x[weight_col]) if x[weight_col].sum() > 0 else np.nan
    ).reset_index(name='weighted_default_rate')

    # Step 3: Merge the results
    summary_df = pd.merge(summary_stats, weighted_rates, on=source_col)

    # Display the summary
    print("Summary of Default Rates by Source:")
    print(summary_df[[source_col, 'row_count', 'unweighted_default_rate', 'sum_weights', 'weighted_default_rate']])
    return summary_df


def create_TTD_data(
    ri_model,
    df_rejected: pd.DataFrame,
    df_accepted: pd.DataFrame,
    ri_features: list,
    modeling_features: list,
    target_col: str = 'default_flag',
    clone_rejected: bool = True
) -> pd.DataFrame:
    """
    Create a Through-the-Door (TTD) dataset using Fuzzy Augmentation.
    
    Applies a trained reject inference (RI) model to rejected applicants to estimate
    their probability of default, then creates weighted copies of the rejected data
    and combines them with the accepted data.
    
    Parameters
    ----------
    ri_model : estimator or None
        Trained scikit-learn compatible model for reject inference.
        If None, skips fuzzy augmentation and uses weight=1 for all.
    
    df_rejected : pd.DataFrame
        DataFrame with rejected applicant data.
    
    df_accepted : pd.DataFrame
        DataFrame with accepted applicant data.
    
    ri_features : list
        Feature names used by ri_model.
    
    modeling_features : list
        Feature names for final TTD dataset.
    
    target_col : str, default='default_flag'
        Name of target column.
    
    clone_rejected : bool, default=True
        If True, creates two copies of rejected data (one for each class).
    
    Returns
    -------
    pd.DataFrame
        Augmented TTD dataset with 'sample_weight' and 'source' columns.
    """
    # Ensure dataframes are copies to avoid modifying originals
    df_rejected_proc = df_rejected.copy()
    df_accepted_proc = df_accepted.copy()

    # Check if ri_model is None - skip fuzzy augmentation if so
    if ri_model is None:
        print("No RI model provided. Creating TTD data with uniform weights (sample_weight=1)...")
        
        # Prepare Accepted Data
        df_accepted_weighted = df_accepted_proc[modeling_features + [target_col]].copy()
        df_accepted_weighted['sample_weight'] = 1.0
        df_accepted_weighted['source'] = 'Accepted'
        
        # Prepare Rejected Data
        df_rejected_weighted = df_rejected_proc[modeling_features].copy()
        df_rejected_weighted['sample_weight'] = 1.0
        df_rejected_weighted[target_col] = None
        df_rejected_weighted['source'] = 'Rejected'
        
        cols_for_ttd = modeling_features + [target_col, 'sample_weight', 'source']
        
        df_ttd = pd.concat(
            [df_accepted_weighted[cols_for_ttd], df_rejected_weighted[cols_for_ttd]],
            ignore_index=True
        )
    else:
        print("Applying RI model to rejected data and calculating weights...")
        
        # 1. Prepare Rejected Data for RI Prediction
        X_rej_ri = df_rejected_proc[ri_features]
        
        # 2. Predict Probabilities for Rejected Applicants
        prob_default_rejected = ri_model.predict_proba(X_rej_ri)[:, 1]
        prob_good_rejected = 1.0 - prob_default_rejected
        
        # 3. Create two weighted copies of rejected data
        # Copy 1: Assumed Bad (target_col = 1)
        df_rejected_bad = df_rejected_proc[modeling_features].copy()
        df_rejected_bad['sample_weight'] = prob_default_rejected
        df_rejected_bad[target_col] = 1
        df_rejected_bad['source'] = 'Rejected'
        
        # Copy 2: Assumed Good (target_col = 0)
        df_rejected_good = df_rejected_proc[modeling_features].copy()
        df_rejected_good['sample_weight'] = prob_good_rejected
        df_rejected_good[target_col] = 0
        df_rejected_good['source'] = 'Rejected'
        
        # 4. Prepare Accepted Data
        df_accepted_weighted = df_accepted_proc[modeling_features + [target_col]].copy()
        df_accepted_weighted['sample_weight'] = 1.0
        df_accepted_weighted['source'] = 'Accepted'
        
        # 5. Define columns for final TTD dataset
        cols_for_ttd = modeling_features + [target_col, 'sample_weight', 'source']
        
        # 6. Concatenate
        if clone_rejected:
            df_ttd = pd.concat(
                [df_accepted_weighted[cols_for_ttd], 
                 df_rejected_bad[cols_for_ttd], 
                 df_rejected_good[cols_for_ttd]],
                ignore_index=True
            )
        else:
            # Use only bad copies
            df_ttd = pd.concat(
                [df_accepted_weighted[cols_for_ttd], df_rejected_bad[cols_for_ttd]],
                ignore_index=True
            )
    
    print(f"TTD dataset created. Shape: {df_ttd.shape}")
    return df_ttd


# =============================================================================
# ADVERSE ACTION CODES
# =============================================================================

def generate_adverse_action_codes(
    df_rejected: pd.DataFrame,
    df_counterfactuals: pd.DataFrame,
    dict_feature_to_action: Dict[str, str],
    list_features_to_compare: List[str]
) -> List[str]:
    """
    Generate adverse action codes based on differences between 
    a rejected applicant and counterfactuals.
    
    Parameters
    ----------
    df_rejected : pd.DataFrame
        DataFrame with the rejected applicant's data (single row).
    
    df_counterfactuals : pd.DataFrame
        DataFrame containing counterfactual data (first row is used).
    
    dict_feature_to_action : dict
        Mapping from feature names to adverse action messages.
    
    list_features_to_compare : list
        List of feature names to compare (in order of importance).
    
    Returns
    -------
    list
        List of adverse action code strings.
    """
    adverse_action_codes = []
    
    for feature in list_features_to_compare:
        original_value = df_rejected[feature].iloc[0]
        counterfactual_value = df_counterfactuals[feature].iloc[0]
        if original_value != counterfactual_value:
            adverse_action_codes.append(dict_feature_to_action[feature])
    
    return adverse_action_codes