from __future__ import annotations

from typing import Iterable, Dict, List


def build_monotone_constraints(
    all_features: Iterable[str],
    constraints_dict: Dict[str, int],
    default_value: int = 0,
) -> List[int]:
    """
    Build a feature-ordered list of monotone constraints.

    Args:
        all_features: Ordered feature list used by the model.
        constraints_dict: Mapping of feature name to constraint value.
        default_value: Value for unconstrained features.

    Returns:
        List of constraints aligned with all_features.
    """
    return [constraints_dict.get(f, default_value) for f in all_features]


if __name__ == "__main__":
    example_features = ["loan_amnt", "dti", "credit_score", "emp_length"]
    example_constraints = {"loan_amnt": 1, "dti": 1, "credit_score": -1, "emp_length": -1}
    print(build_monotone_constraints(example_features, example_constraints))
