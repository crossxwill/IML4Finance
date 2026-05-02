from __future__ import annotations

import pandas as pd
from ydata_profiling import ProfileReport


def generate_eda_report(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    sample_frac: float = 0.1,
    random_state: int = 2025,
) -> None:
    """
    Generate a ydata-profiling report with optional sampling.

    Args:
        df: Input DataFrame.
        title: Report title.
        output_path: HTML output path.
        sample_frac: Fraction of rows to sample.
        random_state: Random seed for sampling.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")

    sample_df = df.sample(frac=sample_frac, random_state=random_state) if 0 < sample_frac < 1 else df

    report = ProfileReport(
        sample_df,
        title=title,
        progress_bar=False,
        duplicates=None,
        interactions=None,
    )
    report.to_file(output_path)
