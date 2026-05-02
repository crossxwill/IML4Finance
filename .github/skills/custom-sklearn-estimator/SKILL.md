---
name: custom-sklearn-estimator
description: Build scikit-learn compatible custom estimators by following the official “rolling your own estimator” rules for __init__, fit/predict, validation, learned attributes, tags, and estimator checks; prerequisite for autogluon-sklearn-wrapper or any sklearn-facing wrappers.
---

# Custom scikit-learn Estimator

## Purpose
Create scikit-learn compatible estimators that work with pipelines, model selection, and validation tooling. This skill codifies the required API patterns for `__init__`, `fit`, prediction/transform methods, learned attributes, and estimator checks.

## Usage
- “rolling your own estimator”
- “custom scikit-learn estimator”
- “build sklearn-compatible class”

## Instructions
1. Choose the estimator type and mixins
   - Use `ClassifierMixin`, `RegressorMixin`, `TransformerMixin`, or `ClusterMixin` as needed, with `BaseEstimator` **last** in the inheritance list.
   - For meta-estimators, ensure sub-estimator params are exposed through `get_params`/`set_params` (handled by `BaseEstimator`).
2. Implement a minimal `__init__`
   - Keyword args with defaults; no validation or logic.
   - Assign each parameter to an attribute with the exact same name.
   - Avoid mutable defaults; do not set attributes with trailing `_` here.
3. Implement `fit`
   - Signature: `fit(self, X, y=None, **kwargs)` and accept `y=None` even for unsupervised estimators.
   - Validate inputs using `validate_data`/`check_array`; ensure `X.shape[0] == y.shape[0]` when supervised.
   - Set learned attributes with trailing `_` (e.g., `coef_`, `classes_`).
   - Return `self` and overwrite learned attributes on every call unless `warm_start=True`.
4. Implement prediction/transform methods
   - Call `check_is_fitted` and validate inputs with `validate_data(..., reset=False)`.
   - Classifiers must use `self.classes_` and return labels, not indices.
   - Transformers must preserve sample count and order.
5. Handle randomness correctly
   - Accept `random_state=None` in `__init__`, store it unmodified.
   - In `fit`, use `check_random_state` and store RNG in `random_state_` if needed later.
6. Optional: tags and `set_output`
   - Implement `__sklearn_tags__` if default tags are not appropriate.
   - For transformers, consider `get_feature_names_out` and `set_output` compatibility.
7. Validate with estimator checks
   - Run `check_estimator` or `parametrize_with_checks` when possible.
   - Use the response checklist in `./templates/estimator-checklist.md` to confirm compliance.
