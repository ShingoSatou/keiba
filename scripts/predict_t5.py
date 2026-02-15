"""
T-5 snapshot 入力の当日推論スクリプト。

mart.t5_runner_snapshot を入力に、推論結果と監査ログをファイル出力する。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from app.services.ev_service import EVService
from scripts.predict import (
    _add_condition_features,
    _add_inrace_features,
    _add_pace_features,
    _join_horse_stats,
    _join_person_stats,
    distance_to_bucket,
    going_to_bucket,
    load_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def detect_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _normalize_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _to_native(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _asof_key(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["asof_ts"], errors="coerce")


def _fetch_snapshot_rows(db: Database, race_date: str, feature_set: str) -> pd.DataFrame:
    sql = """
    SELECT
        s.race_id,
        s.race_date,
        s.track_code,
        s.race_no,
        s.horse_id,
        s.horse_no,
        s.gate,
        s.jockey_id,
        s.trainer_id,
        s.carried_weight,
        s.body_weight_asof,
        s.body_weight_diff_asof,
        s.post_time,
        s.asof_ts,
        s.o1_announce_mmddhhmi,
        s.odds_win_t5,
        s.pop_win_t5,
        s.odds_rank_t5,
        s.odds_snapshot_age_sec,
        s.odds_missing_flag,
        s.wh_announce_mmddhhmi,
        s.event_change_keys,
        s.dm_kbn,
        s.dm_create_time,
        s.tm_kbn,
        s.tm_create_time,
        s.code_version,
        h.horse_name,
        r.surface,
        r.distance_m,
        r.going,
        r.weather,
        r.class_code,
        r.field_size
    FROM mart.t5_runner_snapshot s
    LEFT JOIN core.horse h
      ON h.horse_id = s.horse_id
    LEFT JOIN core.race r
      ON r.race_id = s.race_id
    WHERE s.race_date = %(race_date)s
      AND s.feature_set = %(feature_set)s
    ORDER BY s.race_id, s.asof_ts, s.horse_no
    """
    rows = db.fetch_all(sql, {"race_date": race_date, "feature_set": feature_set})
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["event_change_keys"] = frame["event_change_keys"].apply(_normalize_json)
    frame["asof_ts"] = pd.to_datetime(frame["asof_ts"], errors="coerce")
    return frame


def _merge_rt_mining_features(
    db: Database, frame: pd.DataFrame, race_date: str, feature_set: str
) -> pd.DataFrame:
    if frame.empty:
        return frame

    dm_sql = """
    SELECT
        s.race_id,
        s.horse_no,
        s.asof_ts,
        d.dm_time_x10,
        d.dm_rank
    FROM mart.t5_runner_snapshot s
    LEFT JOIN core.rt_mining_dm d
      ON d.race_id = s.race_id
     AND d.horse_no = s.horse_no
     AND d.data_kbn = s.dm_kbn
     AND (d.data_create_ymd || d.data_create_hm) = s.dm_create_time
    WHERE s.race_date = %(race_date)s
      AND s.feature_set = %(feature_set)s
    """
    tm_sql = """
    SELECT
        s.race_id,
        s.horse_no,
        s.asof_ts,
        t.tm_score,
        t.tm_rank
    FROM mart.t5_runner_snapshot s
    LEFT JOIN core.rt_mining_tm t
      ON t.race_id = s.race_id
     AND t.horse_no = s.horse_no
     AND t.data_kbn = s.tm_kbn
     AND (t.data_create_ymd || t.data_create_hm) = s.tm_create_time
    WHERE s.race_date = %(race_date)s
      AND s.feature_set = %(feature_set)s
    """
    query_params = {"race_date": race_date, "feature_set": feature_set}
    dm_frame = pd.DataFrame(db.fetch_all(dm_sql, query_params))
    tm_frame = pd.DataFrame(db.fetch_all(tm_sql, query_params))

    merged = frame.copy()
    merged["asof_ts_key"] = _asof_key(merged)

    if not dm_frame.empty:
        dm_frame["asof_ts_key"] = _asof_key(dm_frame)
        merged = merged.merge(
            dm_frame[["race_id", "horse_no", "asof_ts_key", "dm_time_x10", "dm_rank"]],
            on=["race_id", "horse_no", "asof_ts_key"],
            how="left",
        )
    else:
        merged["dm_time_x10"] = None
        merged["dm_rank"] = None

    if not tm_frame.empty:
        tm_frame["asof_ts_key"] = _asof_key(tm_frame)
        merged = merged.merge(
            tm_frame[["race_id", "horse_no", "asof_ts_key", "tm_score", "tm_rank"]],
            on=["race_id", "horse_no", "asof_ts_key"],
            how="left",
        )
    else:
        merged["tm_score"] = None
        merged["tm_rank"] = None

    merged["dm_data_kbn"] = merged["dm_kbn"]
    merged["dm_pred_time_sec"] = pd.to_numeric(merged["dm_time_x10"], errors="coerce") / 10.0
    merged["dm_rank"] = pd.to_numeric(merged["dm_rank"], errors="coerce")
    merged["dm_missing_flag"] = merged["dm_pred_time_sec"].isna().astype(int)

    merged["tm_data_kbn"] = merged["tm_kbn"]
    merged["tm_score"] = pd.to_numeric(merged["tm_score"], errors="coerce")
    merged["tm_rank"] = pd.to_numeric(merged["tm_rank"], errors="coerce")
    merged["tm_missing_flag"] = merged["tm_score"].isna().astype(int)
    return merged.drop(columns=["asof_ts_key"], errors="ignore")


def _apply_racewise_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped: list[pd.DataFrame] = []
    for _, race_df in frame.groupby(["race_id", "asof_ts"], sort=False):
        transformed = _add_inrace_features(race_df.copy())
        transformed = _add_pace_features(transformed)
        grouped.append(transformed)
    if not grouped:
        return frame
    return pd.concat(grouped, ignore_index=True)


def _prepare_features(db: Database, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    features = frame.copy()
    features["body_weight"] = features["body_weight_asof"]
    features["body_weight_diff"] = features["body_weight_diff_asof"]
    features["distance_bucket"] = features["distance_m"].apply(
        lambda value: distance_to_bucket(int(value)) if pd.notna(value) else 1000
    )
    features["going_bucket"] = features["going"].apply(
        lambda value: going_to_bucket(int(value)) if pd.notna(value) else going_to_bucket(None)
    )

    race_date = pd.to_datetime(features["race_date"].iloc[0]).date()
    features = _join_horse_stats(db, features, race_date)
    features = _join_person_stats(db, features, race_date)
    features = _apply_racewise_features(features)
    features = _add_condition_features(db, features, race_date)
    return features


def _predict_and_score(
    frame: pd.DataFrame, odds_stale_sec: int, slippage: float, min_prob: float, bet_amount: int
) -> pd.DataFrame:
    if frame.empty:
        return frame

    model, calibrator, feature_names = load_model()
    missing_features = [column for column in feature_names if column not in frame.columns]
    if missing_features:
        logger.warning("推論時に欠損している特徴量: %s", missing_features)

    prediction_frame = frame.copy()
    matrix = prediction_frame.reindex(columns=feature_names)
    _ = model.predict(matrix)
    prediction_frame["p"] = calibrator.predict_proba(matrix)[:, 1]

    prediction_frame["odds_stale_flag"] = (
        pd.to_numeric(prediction_frame["odds_snapshot_age_sec"], errors="coerce")
        .fillna(10**9)
        .astype(float)
        > float(odds_stale_sec)
    ).astype(int)

    ev_service = EVService(slippage=slippage, min_prob=min_prob, bet_amount=bet_amount)
    ev_values: list[float | None] = []
    recommendations: list[str] = []
    bet_amounts: list[int | None] = []

    for row in prediction_frame.itertuples(index=False):
        odds = row.odds_win_t5
        p_win = row.p
        odds_missing = bool(row.odds_missing_flag) or pd.isna(odds)
        odds_stale = int(row.odds_stale_flag) == 1

        if odds_missing:
            ev_values.append(None)
            recommendations.append("skip")
            bet_amounts.append(None)
            continue

        ev_result = ev_service.calculate_ev(float(p_win), float(odds))
        should_buy = ev_result.is_buy and not odds_stale
        ev_values.append(ev_result.ev_profit)
        recommendations.append("buy" if should_buy else "skip")
        bet_amounts.append(ev_service.bet_amount if should_buy else None)

    prediction_frame["ev"] = ev_values
    prediction_frame["recommendation"] = recommendations
    prediction_frame["buy_flag"] = (prediction_frame["recommendation"] == "buy").astype(int)
    prediction_frame["bet_amount"] = bet_amounts
    return prediction_frame


def _race_warnings(race_frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    odds_missing_count = int(race_frame["odds_missing_flag"].fillna(False).astype(bool).sum())
    if odds_missing_count > 0:
        warnings.append(f"odds missing={odds_missing_count}")

    stale_count = int(race_frame["odds_stale_flag"].fillna(0).astype(int).sum())
    if stale_count > 0:
        warnings.append(f"odds stale={stale_count}")

    event_keys = _normalize_json(race_frame.iloc[0].get("event_change_keys"))
    av_horse_nos = event_keys.get("av_horse_nos")
    if isinstance(av_horse_nos, list) and av_horse_nos:
        warnings.append(f"cancel reflected={','.join(str(x) for x in av_horse_nos)}")
    if event_keys.get("tc_announce_mmddhhmi"):
        warnings.append("post time changed")
    if event_keys.get("jc_announce_mmddhhmi"):
        warnings.append("jockey change reflected")

    return warnings


def _build_html(frame: pd.DataFrame, race_date: str, feature_set: str) -> str:
    if frame.empty:
        return (
            "<html><head><meta charset='utf-8'><title>T5 Predictions</title></head>"
            "<body><h1>T5 Predictions</h1><p>No rows.</p></body></html>"
        )

    total_rows = len(frame)
    total_races = frame["race_id"].nunique()
    odds_missing = int(frame["odds_missing_flag"].fillna(False).astype(bool).sum())
    odds_stale = int(frame["odds_stale_flag"].fillna(0).astype(int).sum())
    buy_count = int((frame["recommendation"] == "buy").sum())

    html_parts = [
        "<html><head><meta charset='utf-8'><title>T5 Predictions</title>",
        "<style>body{font-family:sans-serif;padding:20px;} .warn{color:#b45309;font-weight:700;}"
        " .buy{background:#ecfdf5;} table{border-collapse:collapse;margin-bottom:16px;}"
        " th,td{border:1px solid #d1d5db;padding:4px 8px;}</style>",
        "</head><body>",
        f"<h1>T5 Predictions ({race_date}, feature_set={feature_set})</h1>",
        (
            "<p>"
            f"rows={total_rows}, races={total_races}, buys={buy_count}, "
            f"odds_missing={odds_missing}, odds_stale={odds_stale}"
            "</p>"
        ),
    ]

    for (race_id, asof_ts), race_df in frame.groupby(["race_id", "asof_ts"], sort=False):
        race_df = race_df.sort_values("horse_no").copy()
        warnings = _race_warnings(race_df)
        warnings_text = " / ".join(warnings) if warnings else "none"
        html_parts.append(f"<h2>Race {race_id} (asof={asof_ts})</h2>")
        html_parts.append(f"<p class='warn'>warnings: {warnings_text}</p>")
        race_table = race_df[
            [
                "horse_no",
                "horse_name",
                "odds_win_t5",
                "odds_snapshot_age_sec",
                "odds_missing_flag",
                "odds_stale_flag",
                "p",
                "ev",
                "recommendation",
            ]
        ].copy()
        race_table["recommendation"] = race_table["recommendation"].apply(
            lambda value: f"<span class='buy'>{value}</span>" if value == "buy" else value
        )
        html_parts.append(race_table.to_html(index=False, escape=False))

    html_parts.append("</body></html>")
    return "".join(html_parts)


def _build_audit_payload(
    frame: pd.DataFrame,
    race_date: str,
    feature_set: str,
    odds_stale_sec: int,
    slippage: float,
    min_prob: float,
    bet_amount: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_payload: dict[str, Any] = {
        "race_date": race_date,
        "feature_set": feature_set,
        "generated_at": datetime.now().isoformat(),
        "odds_stale_sec": odds_stale_sec,
        "slippage": slippage,
        "min_prob": min_prob,
        "bet_amount": bet_amount,
        "predict_code_version": detect_git_version(),
        "files": {
            "csv": str(output_dir / "t5_predictions.csv"),
            "json": str(output_dir / "t5_predictions.json"),
            "html": str(output_dir / "t5_predictions.html"),
            "audit": str(output_dir / "t5_audit.json"),
        },
        "races": [],
    }

    if frame.empty:
        return run_payload

    races: list[dict[str, Any]] = []
    for (race_id, asof_ts), race_df in frame.groupby(["race_id", "asof_ts"], sort=False):
        row = race_df.iloc[0]
        race_item: dict[str, Any] = {
            "race_id": int(race_id),
            "track_code": int(row["track_code"]) if pd.notna(row["track_code"]) else None,
            "race_no": int(row["race_no"]) if pd.notna(row["race_no"]) else None,
            "post_time": str(row["post_time"]) if pd.notna(row["post_time"]) else None,
            "asof_ts": str(asof_ts),
            "o1_announce_mmddhhmi": _to_native(row.get("o1_announce_mmddhhmi")),
            "wh_announce_mmddhhmi": _to_native(row.get("wh_announce_mmddhhmi")),
            "event_change_keys": _normalize_json(row.get("event_change_keys")),
            "dm_kbn": _to_native(row.get("dm_kbn")),
            "dm_create_time": _to_native(row.get("dm_create_time")),
            "tm_kbn": _to_native(row.get("tm_kbn")),
            "tm_create_time": _to_native(row.get("tm_create_time")),
            "code_version": _to_native(row.get("code_version")),
            "counts": {
                "odds_missing": int(race_df["odds_missing_flag"].fillna(False).astype(bool).sum()),
                "odds_stale": int(race_df["odds_stale_flag"].fillna(0).astype(int).sum()),
                "buy": int((race_df["recommendation"] == "buy").sum()),
            },
        }
        buy_rows = race_df[race_df["recommendation"] == "buy"].sort_values("ev", ascending=False)
        if not buy_rows.empty:
            top = buy_rows.iloc[0]
            race_item["best_buy"] = {
                "horse_no": int(top["horse_no"]) if pd.notna(top["horse_no"]) else None,
                "p": float(top["p"]) if pd.notna(top["p"]) else None,
                "ev": float(top["ev"]) if pd.notna(top["ev"]) else None,
                "stake": int(top["bet_amount"]) if pd.notna(top["bet_amount"]) else None,
            }
        races.append(race_item)
    run_payload["races"] = races
    return run_payload


def run_predict_t5(
    race_date: str,
    feature_set: str,
    output_dir: Path,
    odds_stale_sec: int,
    slippage: float,
    min_prob: float,
    bet_amount: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Database() as db:
        rows = _fetch_snapshot_rows(db, race_date=race_date, feature_set=feature_set)
        if rows.empty:
            logger.warning(
                "snapshot rows are empty (date=%s feature_set=%s)",
                race_date,
                feature_set,
            )
            audit_payload = _build_audit_payload(
                rows,
                race_date,
                feature_set,
                odds_stale_sec,
                slippage,
                min_prob,
                bet_amount,
                output_dir,
            )
            (output_dir / "t5_audit.json").write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return audit_payload

        features = _prepare_features(db, rows)
        features = _merge_rt_mining_features(
            db,
            features,
            race_date=race_date,
            feature_set=feature_set,
        )
        predicted = _predict_and_score(
            features,
            odds_stale_sec=odds_stale_sec,
            slippage=slippage,
            min_prob=min_prob,
            bet_amount=bet_amount,
        )

    output_columns = [
        "race_id",
        "asof_ts",
        "race_no",
        "horse_no",
        "horse_name",
        "odds_win_t5",
        "odds_snapshot_age_sec",
        "odds_missing_flag",
        "odds_stale_flag",
        "p",
        "ev",
        "recommendation",
        "dm_kbn",
        "dm_create_time",
        "tm_kbn",
        "tm_create_time",
        "o1_announce_mmddhhmi",
        "wh_announce_mmddhhmi",
        "code_version",
        "event_change_keys",
    ]
    export_frame = predicted.reindex(columns=output_columns).copy()
    export_frame = export_frame.sort_values(["race_id", "asof_ts", "horse_no"])

    csv_path = output_dir / "t5_predictions.csv"
    json_path = output_dir / "t5_predictions.json"
    html_path = output_dir / "t5_predictions.html"
    audit_path = output_dir / "t5_audit.json"

    export_frame.to_csv(csv_path, index=False, encoding="utf-8")
    export_frame.to_json(json_path, orient="records", force_ascii=False, indent=2)
    html_path.write_text(_build_html(predicted, race_date, feature_set), encoding="utf-8")

    audit_payload = _build_audit_payload(
        predicted,
        race_date,
        feature_set,
        odds_stale_sec,
        slippage,
        min_prob,
        bet_amount,
        output_dir,
    )
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "predictions exported: rows=%s races=%s buys=%s",
        len(export_frame),
        export_frame["race_id"].nunique(),
        int((export_frame["recommendation"] == "buy").sum()),
    )
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="T-5 snapshot 入力の当日推論")
    parser.add_argument("--race-date", required=True, help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--feature-set", required=True, help="snapshot feature_set")
    parser.add_argument("--output-dir", default="", help="出力先 (未指定時は data/ops/<YYYYMMDD>)")
    parser.add_argument(
        "--odds-stale-sec",
        type=int,
        default=900,
        help="オッズ古さ判定しきい値(秒)",
    )
    parser.add_argument("--slippage", type=float, default=0.15, help="スリッページ率")
    parser.add_argument("--min-prob", type=float, default=0.03, help="最低確率閾値")
    parser.add_argument("--bet-amount", type=int, default=500, help="賭け金")
    args = parser.parse_args()

    race_date = args.race_date
    ymd = race_date.replace("-", "")
    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "ops" / ymd)

    audit_payload = run_predict_t5(
        race_date=race_date,
        feature_set=args.feature_set,
        output_dir=output_dir,
        odds_stale_sec=args.odds_stale_sec,
        slippage=args.slippage,
        min_prob=args.min_prob,
        bet_amount=args.bet_amount,
    )
    logger.info("done: %s", json.dumps(audit_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
