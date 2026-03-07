from __future__ import annotations

import pandas as pd

OPERATIONAL_MODE_CHOICES = ("t10_only", "includes_final")
PL_FEATURE_PROFILE_CHOICES = ("stack_default", "meta_default", "raw_legacy")
STACKER_TASK_CHOICES = ("win", "place")
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
BINARY_NON_FEATURE_COLUMNS = [
    "race_id",
    "horse_id",
    "horse_no",
    "year",
    "race_date",
    "race_datetime",
    "t_race",
    "start_time",
    "holdout_year",
    "train_window_years",
    "cv_window_policy",
    "window_definition",
]
BINARY_TE_EXCLUDED_PREFIXES = ("p_", "score_", "c_", "z_", "pl_", "ranker_")
BINARY_TE_EXCLUDED_SUFFIXES = ("_id", "_key", "_dt", "_date", "_time", "_year", "_ymd", "_hm")
BINARY_TE_EXCLUDED_SUBSTRINGS = (
    "_final_",
    "datetime",
    "announce_dt",
    "create_time",
    "create_datetime",
)

STACKER_CONTEXT_FEATURES = [
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
]
STACKER_REQUIRED_PRED_FEATURES_WIN = [
    "p_win_lgbm",
    "p_win_xgb",
    "p_win_cat",
]
STACKER_REQUIRED_PRED_FEATURES_PLACE = [
    "p_place_lgbm",
    "p_place_xgb",
    "p_place_cat",
]
STACKER_WIN_ODDS_FEATURES = [
    "odds_win_t20",
    "odds_win_t15",
    "odds_win_t10",
]
STACKER_PLACE_ODDS_FEATURES = [
    "odds_place_t20_lower",
    "odds_place_t20_upper",
    "place_width_log_ratio_t20",
    "odds_place_t15_lower",
    "odds_place_t15_upper",
    "place_width_log_ratio_t15",
    "odds_place_t10_lower",
    "odds_place_t10_upper",
    "place_width_log_ratio_t10",
]

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
PL_REQUIRED_PRED_FEATURES_STACK = [
    "p_win_stack",
    "p_place_stack",
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
PL_STACK_CORE_FEATURES = [
    "z_win_stack",
    "z_place_stack",
    "place_width_log_ratio",
]
PL_STACK_INTERACTION_FEATURES = [
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
    "z_win_stack_x_z_place_stack",
    "z_win_stack_x_place_width_log_ratio",
    "z_place_stack_x_place_width_log_ratio",
    "z_win_stack_x_field_size",
    "z_place_stack_x_field_size",
    "z_win_stack_x_distance_m",
    "z_place_stack_x_distance_m",
    "z_win_stack_race_centered",
    "z_place_stack_race_centered",
    "place_width_log_ratio_race_centered",
    "z_win_stack_rank_pct",
    "z_place_stack_rank_pct",
    "place_width_log_ratio_rank_pct",
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


def _validate_stacker_task(task: str) -> None:
    if task not in STACKER_TASK_CHOICES:
        raise ValueError(f"Unknown stacker task={task!r}. Expected one of {STACKER_TASK_CHOICES}.")


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


def _is_binary_te_candidate_column(column: str) -> bool:
    lowered = str(column).lower()
    return (
        "target" in lowered
        or lowered.startswith("te_")
        or "_te_" in lowered
        or lowered.endswith("_te")
    )


def _is_safe_binary_te_extra_column(
    column: str,
    *,
    base_contract_columns: set[str],
) -> bool:
    lowered = str(column).lower()
    if column in base_contract_columns:
        return False
    if column in BINARY_NON_FEATURE_COLUMNS:
        return False
    if column in POST_RACE_FORBIDDEN_FEATURES:
        return False
    if column in FORBIDDEN_FINAL_ODDS_FEATURES:
        return False
    if column in BINARY_ENTITY_ID_FEATURES:
        return False
    if not _is_binary_te_candidate_column(column):
        return False
    if lowered.startswith(BINARY_TE_EXCLUDED_PREFIXES):
        return False
    if lowered.endswith(BINARY_TE_EXCLUDED_SUFFIXES):
        return False
    if lowered.startswith("odds_") or "odds_" in lowered:
        return False
    if any(token in lowered for token in BINARY_TE_EXCLUDED_SUBSTRINGS):
        return False
    return True


def get_binary_safe_te_feature_columns(
    frame: pd.DataFrame,
    *,
    operational_mode: str,
    include_entity_ids: bool = False,
) -> list[str]:
    _validate_operational_mode(operational_mode)

    base_contract_columns = [*BINARY_BASE_FEATURES, *BINARY_T10_ODDS_FEATURES]
    if operational_mode == "includes_final":
        base_contract_columns.extend(FINAL_ODDS_BASE_FEATURES)
    if include_entity_ids:
        base_contract_columns.extend(BINARY_ENTITY_ID_FEATURES)

    allowed_set = set(base_contract_columns)
    extras: list[str] = []
    seen: set[str] = set()
    for raw_column in frame.columns:
        column = str(raw_column)
        if column in seen:
            continue
        seen.add(column)
        if _is_safe_binary_te_extra_column(column, base_contract_columns=allowed_set):
            extras.append(column)
    return extras


def get_binary_feature_columns(
    frame: pd.DataFrame,
    include_entity_ids: bool,
    operational_mode: str,
    include_te_features: bool = False,
) -> list[str]:
    _validate_operational_mode(operational_mode)

    cols = [*BINARY_BASE_FEATURES, *BINARY_T10_ODDS_FEATURES]
    if operational_mode == "includes_final":
        cols.extend(FINAL_ODDS_BASE_FEATURES)
    if include_entity_ids:
        cols.extend(BINARY_ENTITY_ID_FEATURES)
    if include_te_features:
        cols.extend(
            get_binary_safe_te_feature_columns(
                frame,
                operational_mode=operational_mode,
                include_entity_ids=include_entity_ids,
            )
        )

    feature_cols = _dedupe_existing(frame, cols)
    validate_feature_contract(
        feature_cols,
        operational_mode=operational_mode,
        stage="binary",
    )
    return feature_cols


def get_stacker_feature_columns(
    frame: pd.DataFrame,
    *,
    task: str,
    operational_mode: str = "t10_only",
) -> list[str]:
    _validate_stacker_task(task)
    _validate_operational_mode(operational_mode)

    if task == "win":
        cols = [
            *STACKER_REQUIRED_PRED_FEATURES_WIN,
            *STACKER_WIN_ODDS_FEATURES,
            *STACKER_CONTEXT_FEATURES,
        ]
    else:
        cols = [
            *STACKER_REQUIRED_PRED_FEATURES_PLACE,
            *STACKER_PLACE_ODDS_FEATURES,
            *STACKER_CONTEXT_FEATURES,
        ]

    feature_cols = _dedupe_existing(frame, cols)
    validate_feature_contract(
        feature_cols,
        operational_mode=operational_mode,
        stage="stacker",
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

    if feature_profile == "stack_default":
        cols = [*PL_STACK_CORE_FEATURES, *PL_STACK_INTERACTION_FEATURES]
    else:
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
    if feature_profile == "stack_default":
        cols = [*PL_REQUIRED_PRED_FEATURES_STACK]
    elif feature_profile == "meta_default":
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
    "BINARY_NON_FEATURE_COLUMNS",
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
    "PL_REQUIRED_PRED_FEATURES_STACK",
    "PL_STACK_CORE_FEATURES",
    "PL_STACK_INTERACTION_FEATURES",
    "PL_T10_ODDS_FEATURES",
    "POST_RACE_FORBIDDEN_FEATURES",
    "STACKER_CONTEXT_FEATURES",
    "STACKER_PLACE_ODDS_FEATURES",
    "STACKER_REQUIRED_PRED_FEATURES_PLACE",
    "STACKER_REQUIRED_PRED_FEATURES_WIN",
    "STACKER_TASK_CHOICES",
    "STACKER_WIN_ODDS_FEATURES",
    "get_binary_feature_columns",
    "get_binary_safe_te_feature_columns",
    "get_pl_feature_columns",
    "get_pl_required_pred_columns",
    "get_stacker_feature_columns",
    "validate_feature_contract",
]
