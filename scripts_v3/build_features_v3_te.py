#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.feature_registry_v3 import get_binary_safe_te_feature_columns
from scripts_v3.v3_common import assert_sorted, hash_files, resolve_path, save_json

logger = logging.getLogger(__name__)

TE_JOIN_KEYS = ["race_id", "horse_id", "horse_no"]
REQUIRED_BASE_V3_COLUMNS = [
    "race_id",
    "horse_id",
    "horse_no",
    "y_win",
    "y_place",
    "finish_pos",
    "p_win_odds_t20_norm",
    "p_win_odds_t15_norm",
    "p_win_odds_t10_norm",
    "d_logit_win_15_20",
    "d_logit_win_10_15",
    "d_logit_win_10_20",
    "place_mid_prob_t20",
    "place_mid_prob_t15",
    "place_mid_prob_t10",
    "d_place_mid_10_20",
    "d_place_width_10_20",
    "place_width_log_ratio",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3 TE features by merging features_v3 with safe TE extra columns."
    )
    parser.add_argument("--base-input", default="data/features_v3.parquet")
    parser.add_argument("--te-source-input", default="data/features_v2_te.parquet")
    parser.add_argument("--output", default="data/features_v3_te.parquet")
    parser.add_argument("--meta-output", default="data/features_v3_te_meta.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _assert_unique_keys(frame: pd.DataFrame, *, label: str) -> None:
    duplicated = frame.duplicated(TE_JOIN_KEYS, keep=False)
    if duplicated.any():
        sample = (
            frame.loc[duplicated, TE_JOIN_KEYS]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(f"{label} has duplicate join keys: sample={sample}")


def _build_coverage(frame: pd.DataFrame, te_feature_columns: list[str]) -> dict[str, float]:
    coverage = {
        "finish_pos_notna_rate": float(frame["finish_pos"].notna().mean()),
        "odds_win_final_notna_rate": float(frame["odds_win_final"].notna().mean()),
        "odds_win_t20_notna_rate": float(frame["odds_win_t20"].notna().mean()),
        "odds_win_t15_notna_rate": float(frame["odds_win_t15"].notna().mean()),
        "odds_win_t10_notna_rate": float(frame["odds_win_t10"].notna().mean()),
        "p_win_odds_t20_norm_notna_rate": float(frame["p_win_odds_t20_norm"].notna().mean()),
        "p_win_odds_t15_norm_notna_rate": float(frame["p_win_odds_t15_norm"].notna().mean()),
        "d_logit_win_15_20_notna_rate": float(frame["d_logit_win_15_20"].notna().mean()),
        "d_logit_win_10_15_notna_rate": float(frame["d_logit_win_10_15"].notna().mean()),
        "d_logit_win_10_20_notna_rate": float(frame["d_logit_win_10_20"].notna().mean()),
        "odds_place_t20_lower_notna_rate": float(frame["odds_place_t20_lower"].notna().mean()),
        "odds_place_t15_lower_notna_rate": float(frame["odds_place_t15_lower"].notna().mean()),
        "odds_place_t10_lower_notna_rate": float(frame["odds_place_t10_lower"].notna().mean()),
        "place_mid_prob_t20_notna_rate": float(frame["place_mid_prob_t20"].notna().mean()),
        "place_mid_prob_t15_notna_rate": float(frame["place_mid_prob_t15"].notna().mean()),
        "place_mid_prob_t10_notna_rate": float(frame["place_mid_prob_t10"].notna().mean()),
        "d_place_mid_10_20_notna_rate": float(frame["d_place_mid_10_20"].notna().mean()),
        "d_place_width_10_20_notna_rate": float(frame["d_place_width_10_20"].notna().mean()),
        "place_width_log_ratio_notna_rate": float(frame["place_width_log_ratio"].notna().mean()),
        "p_win_odds_final_norm_notna_rate": float(frame["p_win_odds_final_norm"].notna().mean()),
        "p_win_odds_t10_norm_notna_rate": float(frame["p_win_odds_t10_norm"].notna().mean()),
    }
    for col in te_feature_columns:
        coverage[f"{col}_notna_rate"] = float(frame[col].notna().mean())
    return coverage


def build_features_v3_te(
    base_df: pd.DataFrame,
    te_source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    _require_columns(base_df, REQUIRED_BASE_V3_COLUMNS, label="base-input")
    _require_columns(te_source_df, TE_JOIN_KEYS, label="te-source-input")
    _assert_unique_keys(base_df, label="base-input")

    te_feature_columns = [
        col
        for col in get_binary_safe_te_feature_columns(
            te_source_df,
            operational_mode="t10_only",
            include_entity_ids=False,
        )
        if col not in base_df.columns
    ]
    if not te_feature_columns:
        raise ValueError("te-source-input does not contain any safe TE extra columns.")

    te_source_cols = [*TE_JOIN_KEYS, *te_feature_columns]
    te_source = te_source_df.loc[:, te_source_cols].copy()
    _assert_unique_keys(te_source, label="te-source-input")

    merged = base_df.merge(
        te_source,
        on=TE_JOIN_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"te-source-input is missing {unmatched} rows from base-input.")

    merged = merged.drop(columns="_merge")
    merged = merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    assert_sorted(merged[["race_id", "horse_no"]].copy())
    return merged, te_feature_columns


def build_features_v3_te_meta_payload(
    frame: pd.DataFrame,
    *,
    base_input_path: Path,
    te_source_input_path: Path,
    output_path: Path,
    te_feature_columns: list[str],
) -> dict[str, object]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_path": str(base_input_path),
        "te_source_input_path": str(te_source_input_path),
        "output_path": str(output_path),
        "operational_default": "t10_only",
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "columns": frame.columns.tolist(),
        "te_feature_columns": te_feature_columns,
        "te_join_keys": [*TE_JOIN_KEYS],
        "contains_final_odds_columns": all(
            col in frame.columns
            for col in [
                "odds_win_final",
                "odds_final_data_kbn",
                "p_win_odds_final_raw",
                "p_win_odds_final_norm",
            ]
        ),
        "contains_t10_odds_columns": all(
            col in frame.columns
            for col in [
                "odds_win_t10",
                "odds_t10_data_kbn",
                "p_win_odds_t10_raw",
                "p_win_odds_t10_norm",
            ]
        ),
        "contains_stacker_timeseries_columns": all(
            col in frame.columns
            for col in [
                "odds_win_t20",
                "odds_win_t15",
                "odds_win_t10",
                "p_win_odds_t20_norm",
                "p_win_odds_t15_norm",
                "p_win_odds_t10_norm",
                "d_logit_win_15_20",
                "d_logit_win_10_15",
                "d_logit_win_10_20",
                "odds_place_t20_lower",
                "odds_place_t20_upper",
                "odds_place_t15_lower",
                "odds_place_t15_upper",
                "odds_place_t10_lower",
                "odds_place_t10_upper",
                "place_mid_prob_t20",
                "place_mid_prob_t15",
                "place_mid_prob_t10",
                "d_place_mid_10_20",
                "d_place_width_10_20",
                "place_width_log_ratio",
            ]
        ),
        "coverage": _build_coverage(frame, te_feature_columns),
        "code_hash": hash_files(
            [
                Path(__file__),
                Path(resolve_path("scripts_v3/feature_registry_v3.py")),
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    base_input_path = resolve_path(args.base_input)
    te_source_input_path = resolve_path(args.te_source_input)
    output_path = resolve_path(args.output)
    meta_output_path = resolve_path(args.meta_output)

    if not base_input_path.exists():
        raise SystemExit(f"base-input not found: {base_input_path}")
    if not te_source_input_path.exists():
        raise SystemExit(f"te-source-input not found: {te_source_input_path}")

    base_df = pd.read_parquet(base_input_path)
    te_source_df = pd.read_parquet(te_source_input_path)
    features_v3_te, te_feature_columns = build_features_v3_te(base_df, te_source_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_v3_te.to_parquet(output_path, index=False)

    meta = build_features_v3_te_meta_payload(
        features_v3_te,
        base_input_path=base_input_path,
        te_source_input_path=te_source_input_path,
        output_path=output_path,
        te_feature_columns=te_feature_columns,
    )
    save_json(meta_output_path, meta)

    logger.info(
        "features_v3_te rows=%s races=%s te_features=%s",
        len(features_v3_te),
        features_v3_te["race_id"].nunique(),
        te_feature_columns,
    )
    logger.info("wrote %s", output_path)
    logger.info("wrote %s", meta_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
