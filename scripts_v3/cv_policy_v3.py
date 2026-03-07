from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_TRAIN_WINDOW_YEARS = 4
DEFAULT_CV_WINDOW_POLICY = "fixed_sliding"
DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS = 2
DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS = 4


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_years: tuple[int, ...]
    valid_year: int


def make_window_definition(train_window_years: int) -> str:
    return f"train = previous {int(train_window_years)} years only, valid = current year"


def make_capped_expanding_window_definition(min_window_years: int, max_window_years: int) -> str:
    return (
        "train = capped expanding years "
        f"(min={int(min_window_years)}, max={int(max_window_years)}), "
        "valid = next year"
    )


def build_fixed_window_year_folds(
    years: list[int],
    *,
    window_years: int = DEFAULT_TRAIN_WINDOW_YEARS,
    holdout_year: int,
) -> list[FoldSpec]:
    """Build fixed-length sliding yearly folds with holdout exclusion."""
    if int(window_years) <= 0:
        raise ValueError("window_years must be > 0")

    trainable_years = sorted({int(year) for year in years if int(year) < int(holdout_year)})
    folds: list[FoldSpec] = []
    for idx in range(int(window_years), len(trainable_years)):
        folds.append(
            FoldSpec(
                fold_id=len(folds) + 1,
                train_years=tuple(trainable_years[idx - int(window_years) : idx]),
                valid_year=int(trainable_years[idx]),
            )
        )
    return folds


def build_capped_expanding_year_folds(
    years: list[int],
    *,
    min_window_years: int = DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    max_window_years: int = DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    holdout_year: int,
) -> list[FoldSpec]:
    if int(min_window_years) <= 0:
        raise ValueError("min_window_years must be > 0")
    if int(max_window_years) < int(min_window_years):
        raise ValueError("max_window_years must be >= min_window_years")

    trainable_years = sorted({int(year) for year in years if int(year) < int(holdout_year)})
    folds: list[FoldSpec] = []
    for idx in range(int(min_window_years), len(trainable_years)):
        window = min(int(max_window_years), idx)
        folds.append(
            FoldSpec(
                fold_id=len(folds) + 1,
                train_years=tuple(trainable_years[idx - window : idx]),
                valid_year=int(trainable_years[idx]),
            )
        )
    return folds


def select_recent_window_years(
    years: list[int],
    *,
    train_window_years: int = DEFAULT_TRAIN_WINDOW_YEARS,
    holdout_year: int | None = None,
) -> list[int]:
    usable = sorted(
        {int(year) for year in years if holdout_year is None or int(year) < int(holdout_year)}
    )
    if not usable:
        return []
    window = int(train_window_years)
    return usable[-window:] if len(usable) > window else usable


def build_cv_policy_payload(
    folds: list[FoldSpec],
    *,
    train_window_years: int = DEFAULT_TRAIN_WINDOW_YEARS,
    holdout_year: int,
    cv_window_policy: str = DEFAULT_CV_WINDOW_POLICY,
    window_definition: str | None = None,
) -> dict[str, object]:
    return {
        "cv_window_policy": str(cv_window_policy),
        "train_window_years": int(train_window_years),
        "valid_years": [int(fold.valid_year) for fold in folds],
        "holdout_year": int(holdout_year),
        "window_definition": (
            str(window_definition)
            if window_definition is not None
            else make_window_definition(int(train_window_years))
        ),
    }


def attach_cv_policy_columns(
    frame: pd.DataFrame,
    *,
    train_window_years: int = DEFAULT_TRAIN_WINDOW_YEARS,
    holdout_year: int,
    cv_window_policy: str = DEFAULT_CV_WINDOW_POLICY,
    window_definition: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["cv_window_policy"] = str(cv_window_policy)
    out["train_window_years"] = int(train_window_years)
    out["holdout_year"] = int(holdout_year)
    out["window_definition"] = (
        str(window_definition)
        if window_definition is not None
        else make_window_definition(int(train_window_years))
    )
    return out


__all__ = [
    "DEFAULT_CV_WINDOW_POLICY",
    "DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS",
    "DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS",
    "DEFAULT_TRAIN_WINDOW_YEARS",
    "FoldSpec",
    "attach_cv_policy_columns",
    "build_capped_expanding_year_folds",
    "build_cv_policy_payload",
    "build_fixed_window_year_folds",
    "make_capped_expanding_window_definition",
    "make_window_definition",
    "select_recent_window_years",
]
