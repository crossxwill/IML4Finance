from __future__ import annotations

import pandas as pd


def create_ttd_data(
    ri_model,
    df_rejected: pd.DataFrame,
    df_accepted: pd.DataFrame,
    ri_features: list[str],
    modeling_features: list[str],
    target_col: str,
    weight_col: str = "sample_weight",
    source_col: str = "source",
) -> pd.DataFrame:
    """
    Create a Through-the-Door dataset using fuzzy augmentation.

    Args:
        ri_model: Fitted model with predict_proba.
        df_rejected: Rejected applicants data.
        df_accepted: Accepted applicants data.
        ri_features: Features used to score rejected data.
        modeling_features: Features to keep in final dataset.
        target_col: Target column name.
        weight_col: Sample weight column name.
        source_col: Source column name.

    Returns:
        Augmented TTD DataFrame.
    """
    pd_rejected = df_rejected.copy()
    pd_accepted = df_accepted.copy()

    # Predict PD on rejected
    pd_scores = ri_model.predict_proba(pd_rejected[ri_features])[:, 1]

    # Rejected as bad
    rejected_bad = pd_rejected.copy()
    rejected_bad[target_col] = 1
    rejected_bad[weight_col] = pd_scores
    rejected_bad[source_col] = "rejected_assumed_bad"

    # Rejected as good
    rejected_good = pd_rejected.copy()
    rejected_good[target_col] = 0
    rejected_good[weight_col] = 1 - pd_scores
    rejected_good[source_col] = "rejected_assumed_good"

    # Accepted data
    accepted = pd_accepted.copy()
    accepted[weight_col] = 1.0
    accepted[source_col] = "accepted"

    cols_keep = modeling_features + [target_col, weight_col, source_col]

    ttd = pd.concat(
        [accepted[cols_keep], rejected_bad[cols_keep], rejected_good[cols_keep]],
        axis=0,
        ignore_index=True,
    )

    return ttd
