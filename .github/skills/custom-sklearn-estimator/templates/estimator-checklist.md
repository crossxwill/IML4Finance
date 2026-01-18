# Custom Estimator Checklist

- [ ] `__init__` has only keyword args with defaults
- [ ] `__init__` assigns parameters directly to attributes (no validation)
- [ ] `fit(self, X, y=None, **kwargs)` returns `self`
- [ ] Input validation via `validate_data`/`check_array`
- [ ] Learned attributes use trailing `_`
- [ ] `n_features_in_` / `feature_names_in_` set via `validate_data`
- [ ] `check_is_fitted` used in `predict`/`transform`
- [ ] Classifiers store `classes_` and return labels
- [ ] Randomness uses `random_state` + `check_random_state`
- [ ] Optional tags and `set_output` are considered
- [ ] `check_estimator` passes (or checks reviewed)
