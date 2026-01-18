# Monotonic Constraints Summary

## Constraint map
- `loan_amnt`: +1
- `dti`: +1
- `credit_score`: -1
- `emp_length`: -1

## Model mapping
- GBM style models use the feature-ordered list
- CatBoost uses the dictionary

## Checks
- Re-run PDP or ICE to verify monotonic behavior
