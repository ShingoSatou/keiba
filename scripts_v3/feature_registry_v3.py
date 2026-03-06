from __future__ import annotations

import pandas as pd

OPERATIONAL_MODE_CHOICES = ("t10_only", "includes_final")
PL_FEATURE_PROFILE_CHOICES = ("meta_default", "raw_legacy")
FEATURE_MANIFEST_VERSION = 1

BINARY_BASE_FEATURES = [
    "track_code",
    "surface",
    "distance_m",
    "going",
    "weather",
    "field_size",
    "grade_code",
    "race_type_code",
    "weight_type_code",
    "condition_code_min_age",
    "age",
    "sex",
    "carried_weight",
    "body_weight",
    "body_weight_diff",
    "is_jockey_change",
    "days_since_lag1",
    "lag1_distance_diff",
    "lag1_course_type_match",
    "lag1_finish_pos",
    "lag2_finish_pos",
    "lag3_finish_pos",
    "lag1_speed_index",
    "lag2_speed_index",
    "lag3_speed_index",
    "lag1_up3_index",
    "lag2_up3_index",
    "lag3_up3_index",
    "apt_same_distance_top3_rate_2y",
    "apt_same_going_top3_rate_2y",
    "meta_dm_time_x10",
    "meta_dm_rank",
    "meta_tm_score",
    "meta_tm_rank",
    "jockey_top3_rate_6m",
    "trainer_top3_rate_6m",
    "rel_lag1_speed_index_z",
    "rel_lag1_speed_index_rank",
    "rel_lag1_speed_index_pct",
    "rel_carried_weight_z",
    "rel_jockey_top3_rate_z",
    "rel_meta_tm_score_z",
]
BINARY_T10_ODDS_FEATURES = [
    "odds_win_t10",
    "odds_t10_data_kbn",
    "p_win_odds_t10_raw",
    "p_win_odds_t10_norm",
]
BINARY_ENTITY_ID_FEATURES = ["jockey_key", "trainer_key"]

PL_REQUIRED_PRED_FEATURES_RAW_LEGACY = [
    "p_win_lgbm",
    "p_win_xgb",
    "p_win_cat",
    "p_place_lgbm",
    "p_place_xgb",
    "p_place_cat",
]
PL_REQUIRED_PRED_FEATURES_META = [
    "p_win_meta",
    "p_place_meta",
]
PL_REQUIRED_PRED_FEATURES = PL_REQUIRED_PRED_FEATURES_RAW_LEGACY
PL_META_DEFAULT_ODDS_FEATURES = ["p_win_odds_t10_norm"]
PL_T10_ODDS_FEATURES = [
    "odds_win_t10",
    "odds_t10_data_kbn",
    "p_win_odds_t10_raw",
    "p_win_odds_t10_norm",
]
PL_CONTEXT_FEATURES_SMALL = [
    "field_size",
    "surface",
    "distance_m",
    "going",
    "apt_same_distance_top3_rate_2y",
    "apt_same_going_top3_rate_2y",
    "rel_lag1_speed_index_z",
    "rel_meta_tm_score_z",
]

FINAL_ODDS_BASE_FEATURES = [
    "odds_win_final",
    "odds_final_data_kbn",
    "p_win_odds_final_raw",
    "p_win_odds_final_norm",
]
FORBIDDEN_FINAL_ODDS_FEATURES = [
    "odds_win_final",
    "odds_final_data_kbn",
    "odds_final_announce_dt",
    "p_win_odds_final_raw",
    "p_win_odds_final_norm",
    "p_win_odds_final_norm_cal_isotonic",
    "p_win_odds_final_norm_cal_logreg",
]
POST_RACE_FORBIDDEN_FEATURES = [
    "target_label",
    "finish_pos",
    "y_win",
    "y_place",
    "y_top3",
    "fold_id",
    "valid_year",
    "pl_score",
    "p_top3",
]


def _validate_operational_mode(operational_mode: str) -> None:
    if operational_mode not in OPERATIONAL_MODE_CHOICES:
        raise ValueError(
            f"Unknown operational_mode={operational_mode!r}. "
            f"Expected one of {OPERATIONAL_MODE_CHOICES}."
        )


def _validate_pl_feature_profile(feature_profile: str) -> None:
    if feature_profile not in PL_FEATURE_PROFILE_CHOICES:
        raise ValueError(
            f"Unknown feature_profile={feature_profile!r}. "
            f"Expected one of {PL_FEATURE_PROFILE_CHOICES}."
        )


def _dedupe_existing(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    existing = set(map(str, frame.columns))
    deduped: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col in seen or col not in existing:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def get_binary_feature_columns(
    frame: pd.DataFrame,
    include_entity_ids: bool,
    operational_mode: str,
) -> list[str]:
    _validate_operational_mode(operational_mode)

    cols = [*BINARY_BASE_FEATURES, *BINARY_T10_ODDS_FEATURES]
    if operational_mode == "includes_final":
        cols.extend(FINAL_ODDS_BASE_FEATURES)
    if include_entity_ids:
        cols.extend(BINARY_ENTITY_ID_FEATURES)

    feature_cols = _dedupe_existing(frame, cols)
    validate_feature_contract(
        feature_cols,
        operational_mode=operational_mode,
        stage="binary",
    )
    return feature_cols


def get_pl_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_profile: str,
    required_pred_cols: list[str],
    include_context: bool,
    include_final_odds: bool,
    operational_mode: str,
) -> list[str]:
    _validate_operational_mode(operational_mode)
    _validate_pl_feature_profile(feature_profile)

    if feature_profile == "meta_default":
        odds_cols = PL_META_DEFAULT_ODDS_FEATURES
    else:
        odds_cols = PL_T10_ODDS_FEATURES

    cols = [*required_pred_cols, *odds_cols]
    if include_context:
        cols.extend(PL_CONTEXT_FEATURES_SMALL)
    if feature_profile == "raw_legacy" and (
        include_final_odds or operational_mode == "includes_final"
    ):
        cols.extend(FINAL_ODDS_BASE_FEATURES)

    feature_cols = _dedupe_existing(frame, cols)
    validate_feature_contract(
        feature_cols,
        operational_mode=operational_mode,
        stage="pl",
    )
    return feature_cols


def get_pl_required_pred_columns(
    feature_profile: str,
    odds_cal_cols: list[str] | None = None,
    include_calibrated_odds_features: bool = False,
) -> list[str]:
    _validate_pl_feature_profile(feature_profile)
    cols: list[str]
    if feature_profile == "meta_default":
        cols = [*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES]
    else:
        cols = [*PL_REQUIRED_PRED_FEATURES_RAW_LEGACY]
        if include_calibrated_odds_features and odds_cal_cols:
            cols.extend(list(odds_cal_cols))
    deduped: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def validate_feature_contract(
    feature_cols: list[str],
    operational_mode: str,
    stage: str,
) -> None:
    _validate_operational_mode(operational_mode)

    seen: set[str] = set()
    duplicates: list[str] = []
    for col in feature_cols:
        if col in seen:
            duplicates.append(col)
            continue
        seen.add(col)
    if duplicates:
        raise ValueError(f"{stage}: duplicate feature columns: {sorted(set(duplicates))}")

    post_race = [col for col in feature_cols if col in POST_RACE_FORBIDDEN_FEATURES]
    if post_race:
        raise ValueError(f"{stage}: post-race forbidden features detected: {post_race}")

    if operational_mode == "t10_only":
        forbidden_final = [
            col for col in feature_cols if col in FORBIDDEN_FINAL_ODDS_FEATURES or "_final_" in col
        ]
        if forbidden_final:
            raise ValueError(
                f"{stage}: final-odds features forbidden in t10_only mode: {forbidden_final}"
            )


__all__ = [
    "BINARY_BASE_FEATURES",
    "BINARY_ENTITY_ID_FEATURES",
    "BINARY_T10_ODDS_FEATURES",
    "FEATURE_MANIFEST_VERSION",
    "FINAL_ODDS_BASE_FEATURES",
    "FORBIDDEN_FINAL_ODDS_FEATURES",
    "OPERATIONAL_MODE_CHOICES",
    "PL_CONTEXT_FEATURES_SMALL",
    "PL_FEATURE_PROFILE_CHOICES",
    "PL_META_DEFAULT_ODDS_FEATURES",
    "PL_REQUIRED_PRED_FEATURES",
    "PL_REQUIRED_PRED_FEATURES_META",
    "PL_REQUIRED_PRED_FEATURES_RAW_LEGACY",
    "PL_T10_ODDS_FEATURES",
    "POST_RACE_FORBIDDEN_FEATURES",
    "get_binary_feature_columns",
    "get_pl_feature_columns",
    "get_pl_required_pred_columns",
    "validate_feature_contract",
]
