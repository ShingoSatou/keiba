#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.odds_v3_common import assert_t10_no_future_reference  # noqa: E402
from scripts_v3.pl_v3_common import (  # noqa: E402
    estimate_p_top3_by_race,
    estimate_p_wide_by_race,
    predict_linear_scores,
)
from scripts_v3.train_binary_v3_common import coerce_feature_matrix, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MC_SAMPLES = 10_000
DEFAULT_SEED = 42

DEFAULT_WIN_METAS = [
    "models/win_lgbm_bundle_meta_v3.json",
    "models/win_xgb_bundle_meta_v3.json",
    "models/win_cat_bundle_meta_v3.json",
]
DEFAULT_PLACE_METAS = [
    "models/place_lgbm_bundle_meta_v3.json",
    "models/place_xgb_bundle_meta_v3.json",
    "models/place_cat_bundle_meta_v3.json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict single-race p_top3 (and optional p_wide) with v3 operational t10 path."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet/CSV with one-race horse-level rows.",
    )
    parser.add_argument("--pl-model", default="models/pl_v3_recent_window.joblib")
    parser.add_argument("--odds-calibrator", default="models/odds_win_calibrators_v3.pkl")
    parser.add_argument("--win-metas", default=",".join(DEFAULT_WIN_METAS))
    parser.add_argument("--place-metas", default=",".join(DEFAULT_PLACE_METAS))
    parser.add_argument(
        "--skip-base-model-inference",
        action="store_true",
        help="If set, assumes p_win_* / p_place_* already exist in --input.",
    )
    parser.add_argument("--output", default="data/predictions/race_v3_pred.parquet")
    parser.add_argument("--wide-output", default="data/predictions/race_v3_wide.parquet")
    parser.add_argument("--emit-wide", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)

    required = {"race_id", "horse_no"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}")

    out = frame.copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)

    race_ids = out["race_id"].unique().tolist()
    if len(race_ids) != 1:
        raise SystemExit(f"--input must contain exactly one race_id. got={race_ids[:5]}")
    return out


def _load_lgbm_model(path: Path):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("lightgbm is required to run predict_race_v3.py") from exc
    return lgb.Booster(model_file=str(path))


def _load_xgb_model(path: Path):
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("xgboost is required. Run `uv sync --extra xgboost`.") from exc
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def _load_cat_model(path: Path):
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("catboost is required. Run `uv sync --extra catboost`.") from exc
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _predict_with_meta(
    frame: pd.DataFrame,
    *,
    meta_path: Path,
) -> tuple[str, np.ndarray]:
    if not meta_path.exists():
        raise SystemExit(f"model meta not found: {meta_path}")
    meta = _load_json(meta_path)

    task = str(meta.get("task", ""))
    model_name = str(meta.get("model", ""))
    pred_col = str(meta.get("pred_col", ""))
    if not pred_col:
        pred_col = f"p_{task}_{model_name}"

    feature_cols = list(map(str, meta.get("feature_columns", [])))
    if not feature_cols:
        raise SystemExit(f"feature_columns missing in {meta_path}")
    output_paths = meta.get("output_paths", {})
    model_path_raw = output_paths.get("main_model")
    if not model_path_raw:
        raise SystemExit(f"main_model path missing in {meta_path}")
    model_path = Path(model_path_raw).resolve()
    if not model_path.exists():
        raise SystemExit(f"main_model not found: {model_path}")

    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise SystemExit(f"Input missing model features for {meta_path.name}: {missing[:10]}")

    x = coerce_feature_matrix(frame, feature_cols)
    x_np = x.to_numpy(dtype=float)

    if model_name == "lgbm":
        model = _load_lgbm_model(model_path)
        pred = model.predict(x_np)
    elif model_name == "xgb":
        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise SystemExit("xgboost is required. Run `uv sync --extra xgboost`.") from exc
        booster = _load_xgb_model(model_path)
        pred = booster.predict(xgb.DMatrix(x_np, feature_names=feature_cols))
    elif model_name == "cat":
        model = _load_cat_model(model_path)
        pred = model.predict_proba(x_np)[:, 1]
    else:
        raise SystemExit(f"Unsupported model in meta {meta_path}: {model_name}")

    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 1e-8, 1.0 - 1e-8)
    return pred_col, pred


def _apply_base_models(
    frame: pd.DataFrame,
    *,
    win_meta_paths: list[Path],
    place_meta_paths: list[Path],
) -> pd.DataFrame:
    out = frame.copy()
    for meta_path in [*win_meta_paths, *place_meta_paths]:
        pred_col, pred = _predict_with_meta(out, meta_path=meta_path)
        out[pred_col] = pred
    return out


def _predict_calibrator(model: Any, method: str, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model.get("constant_prob", 0.0)), dtype=float)
    if method == "logreg":
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return model.predict(x)
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_odds_calibrators(frame: pd.DataFrame, *, model_path: Path) -> pd.DataFrame:
    if not model_path.exists():
        return frame

    bundle = joblib.load(model_path)
    models = bundle.get("models", {})
    if not isinstance(models, dict) or not models:
        return frame

    out = frame.copy()
    for pred_col, item in models.items():
        if pred_col in out.columns:
            continue
        if not isinstance(item, dict):
            continue
        score_col = str(item.get("score_col", ""))
        method = str(item.get("method", ""))
        model = item.get("model")
        if not score_col or score_col not in out.columns:
            continue
        x = pd.to_numeric(out[score_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x)
        pred = np.full(len(x), np.nan, dtype=float)
        if valid.any():
            pred_valid = _predict_calibrator(model, method, x[valid])
            pred_valid = np.clip(np.asarray(pred_valid, dtype=float), 1e-8, 1.0 - 1e-8)
            pred[valid] = pred_valid
        out[pred_col] = pred
    return out


def _validate_operational_t10(feature_cols: list[str]) -> None:
    forbidden = [col for col in feature_cols if "_final_" in col]
    if forbidden:
        raise SystemExit(
            f"Operational t10 path forbids final-odds features in PL model. Found: {forbidden[:10]}"
        )


def _score_with_pl(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    required_keys = {"feature_columns", "weights", "preprocess"}
    missing_keys = sorted(required_keys - set(artifact.keys()))
    if missing_keys:
        raise SystemExit(f"Invalid PL artifact: missing keys {missing_keys}")

    feature_cols = list(map(str, artifact["feature_columns"]))
    _validate_operational_t10(feature_cols)

    missing_features = [col for col in feature_cols if col not in frame.columns]
    if missing_features:
        raise SystemExit(f"Input missing PL features: {missing_features[:10]}")

    preprocess = artifact["preprocess"]
    median_map = preprocess["median"]
    mean_map = preprocess["mean"]
    std_map = preprocess["std"]

    mat = frame[feature_cols].copy()
    for col in feature_cols:
        mat[col] = pd.to_numeric(mat[col], errors="coerce")
        mat[col] = mat[col].fillna(float(median_map[col]))
        scale = float(std_map[col]) if float(std_map[col]) != 0.0 else 1.0
        mat[col] = (mat[col] - float(mean_map[col])) / scale

    x = mat.to_numpy(dtype=float)
    weights = np.asarray(artifact["weights"], dtype=float)
    if x.shape[1] != weights.shape[0]:
        raise SystemExit(
            f"PL weight dimension mismatch: features={x.shape[1]} weights={weights.shape[0]}"
        )
    scores = predict_linear_scores(x, weights)

    out = frame.copy()
    out["pl_score"] = np.asarray(scores, dtype=float)
    return out


def _attach_cv_policy_from_artifact(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    cv_policy = artifact.get("cv_policy", {})
    if not isinstance(cv_policy, dict) or not cv_policy:
        return frame

    out = frame.copy()
    for key in ("cv_window_policy", "train_window_years", "holdout_year", "window_definition"):
        if key in cv_policy:
            out[key] = cv_policy[key]
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    if int(args.mc_samples) <= 0:
        raise SystemExit("--mc-samples must be > 0")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    wide_output_path = resolve_path(args.wide_output)
    pl_model_path = resolve_path(args.pl_model)
    odds_cal_path = resolve_path(args.odds_calibrator)
    win_meta_paths = [resolve_path(path) for path in _parse_csv(args.win_metas)]
    place_meta_paths = [resolve_path(path) for path in _parse_csv(args.place_metas)]

    frame = _load_input(input_path)
    assert_t10_no_future_reference(frame)

    if not bool(args.skip_base_model_inference):
        frame = _apply_base_models(
            frame,
            win_meta_paths=win_meta_paths,
            place_meta_paths=place_meta_paths,
        )
    frame = _apply_odds_calibrators(frame, model_path=odds_cal_path)

    if not pl_model_path.exists():
        raise SystemExit(f"pl model not found: {pl_model_path}")
    artifact = joblib.load(pl_model_path)
    scored = _score_with_pl(frame, artifact)

    p_top3 = estimate_p_top3_by_race(
        scored[["race_id", "horse_no", "pl_score"]],
        score_col="pl_score",
        mc_samples=int(args.mc_samples),
        seed=int(args.seed),
        top_k=3,
    )
    scored = scored.merge(p_top3, on=["race_id", "horse_no"], how="left")
    if "race_date" in scored.columns:
        race_year = pd.to_datetime(scored["race_date"], errors="coerce").dt.year
        if race_year.notna().all():
            scored["valid_year"] = race_year.astype(int)
    scored = _attach_cv_policy_from_artifact(scored, artifact)
    scored = scored.sort_values(["race_id", "horse_no"], kind="mergesort")

    keep_cols = [
        c
        for c in [
            "race_id",
            "horse_id",
            "horse_no",
            "race_date",
            "odds_win_t10",
            "p_win_odds_t10_raw",
            "p_win_odds_t10_norm",
            "p_win_lgbm",
            "p_win_xgb",
            "p_win_cat",
            "p_place_lgbm",
            "p_place_xgb",
            "p_place_cat",
            "pl_score",
            "p_top3",
            "valid_year",
            "cv_window_policy",
            "train_window_years",
            "holdout_year",
            "window_definition",
        ]
        if c in scored.columns
    ]
    pred_out = scored[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_parquet(output_path, index=False)
    logger.info("wrote %s", output_path)

    if bool(args.emit_wide):
        wide = estimate_p_wide_by_race(
            scored[["race_id", "horse_no", "pl_score"]],
            score_col="pl_score",
            mc_samples=int(args.mc_samples),
            seed=int(args.seed),
            top_k=3,
        )
        if "valid_year" in scored.columns:
            year_map = (
                scored[["race_id", "valid_year"]]
                .drop_duplicates(["race_id"])
                .set_index("race_id")["valid_year"]
                .to_dict()
            )
            wide["valid_year"] = wide["race_id"].map(year_map).astype(int)
        wide = _attach_cv_policy_from_artifact(wide, artifact)
        wide_output_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(wide_output_path, index=False)
        logger.info("wrote %s", wide_output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
