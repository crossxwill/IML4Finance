# AutoGluon sklearn wrapper skeleton

Use this skeleton when creating or updating a wrapper.

- Inherit from `ClassifierMixin` and `BaseEstimator` (mixin first)
- Expose `__sklearn_is_fitted__()`
- Set `feature_names_in_` and `n_features_in_` in `fit()`

```python
class AutoGluonSklearnWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, label, predictor_args=None, fit_args=None):
        self.label = label
        self.predictor_args = predictor_args
        self.fit_args = fit_args

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    def fit(self, X, y, sample_weight=None):
        # X, y = validate_data(self, X, y)
        # self.classes_ = unique_labels(y)
        # self.n_features_in_ and feature_names_in_ set by validate_data
        # build DataFrame using feature_names_in_ or generated names
        # add label and optional sample_weight column
        # train TabularPredictor with copied predictor_args/fit_args
        # set self.predictor, self.feature_names_, self.is_fitted_
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # check_is_fitted and validate feature names
        # X = validate_data(self, X, reset=False)
        return self.predictor.predict(...).values

    def predict_proba(self, X):
        # check_is_fitted and validate feature names
        # X = validate_data(self, X, reset=False)
        return self.predictor.predict_proba(...).values
```
