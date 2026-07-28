"""Result serialisation — one summary row plus the full CV grid."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def save_results(
    summary_dict: Dict,
    grid_df: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> None:
    """Write the one-row summary and the full hyperparameter grid to CSV."""
    pd.DataFrame([summary_dict]).to_csv(output_paths["summary_csv"], index=False)
    grid_df.to_csv(output_paths["grid_csv"], index=False)
    print(f"  [Results] summary -> {output_paths['summary_csv']}")
    print(f"  [Results] grid    -> {output_paths['grid_csv']}")
