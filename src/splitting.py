"""Leakage-resistant resampling for hierarchical biodegradation data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


IndexPair = tuple[np.ndarray, np.ndarray]


def curve_group_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = "curve_id",
) -> list[IndexPair]:
    """Return exhaustive folds in which each curve is kept intact."""
    groups = data[group_column].to_numpy()
    unique_groups = pd.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("n_splits must be between 2 and the number of curves")
    splitter = GroupKFold(n_splits=n_splits)
    return [(train, test) for train, test in splitter.split(data, groups=groups)]


def leave_one_study_out_folds(
    data: pd.DataFrame,
    study_column: str = "study_id",
) -> list[IndexPair]:
    """Return one fold for every study, ordered by study identifier."""
    studies = data[study_column].to_numpy()
    folds: list[IndexPair] = []
    for study in sorted(pd.unique(studies)):
        test = np.flatnonzero(studies == study)
        train = np.flatnonzero(studies != study)
        folds.append((train, test))
    return folds

