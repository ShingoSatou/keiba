"""
学習データ品質チェック

train.parquet を読み込み、学習/検証/テスト分割ごとのデータ数・ラベル整合性・欠損率などを出力する。

使用方法:
    uv run python scripts/check_dataset_quality.py --input data/train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import FEATURE_COLS, TARGET_COL, _split_by_race_id


@dataclass
class SplitStats:
    name: str
    rows: int
    races: int
    positives: int
    pos_rate: float
    date_min: str
    date_max: str
    winner_races_0: int
    winner_races_1: int
    winner_races_2: int
    winner_races_gt2: int
    races_lt_min_horses: int
    placeholder_rate: float


def _fmt_date(value) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return str(pd.to_datetime(value).date())
    except Exception:
        return str(value)


def _calc_split_stats(name: str, df: pd.DataFrame, min_horses: int) -> SplitStats:
    if df.empty:
        return SplitStats(
            name=name,
            rows=0,
            races=0,
            positives=0,
            pos_rate=float("nan"),
            date_min="NA",
            date_max="NA",
            winner_races_0=0,
            winner_races_1=0,
            winner_races_2=0,
            winner_races_gt2=0,
            races_lt_min_horses=0,
            placeholder_rate=float("nan"),
        )

    y = df[TARGET_COL].astype(int)
    winner_counts = df.groupby("race_id")[TARGET_COL].sum()
    sizes = df.groupby("race_id").size()
    placeholder = (df["surface"] == 0) | (df["distance_m"] == 0)

    return SplitStats(
        name=name,
        rows=int(len(df)),
        races=int(df["race_id"].nunique()),
        positives=int(y.sum()),
        pos_rate=float(y.mean()),
        date_min=_fmt_date(df["race_date"].min()),
        date_max=_fmt_date(df["race_date"].max()),
        winner_races_0=int((winner_counts == 0).sum()),
        winner_races_1=int((winner_counts == 1).sum()),
        winner_races_2=int((winner_counts == 2).sum()),
        winner_races_gt2=int((winner_counts > 2).sum()),
        races_lt_min_horses=int((sizes < min_horses).sum()),
        placeholder_rate=float(placeholder.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="学習データ品質チェック")
    parser.add_argument("--input", type=str, default="data/train.parquet", help="入力parquet")
    parser.add_argument("--test-size", type=float, default=0.2, help="テスト割合")
    parser.add_argument("--es-val-size", type=float, default=0.1, help="ES用検証割合")
    parser.add_argument("--min-horses", type=int, default=5, help="レースあたり最低頭数")
    parser.add_argument("--json-output", type=str, default=None, help="集計結果のJSON出力先")
    parser.add_argument("--gate", action="store_true", help="品質ゲート判定を有効化")
    parser.add_argument("--max-placeholder-rate", type=float, default=0.001)
    parser.add_argument("--max-dist-lt800-rate", type=float, default=0.0)
    parser.add_argument("--max-dist-lt100-rate", type=float, default=0.0)
    parser.add_argument("--max-going-zero-rate", type=float, default=0.0)
    parser.add_argument("--max-field-size-zero-rate", type=float, default=0.0)
    parser.add_argument("--max-class-code-missing-rate", type=float, default=0.0)
    parser.add_argument("--max-n-runs5-missing-rate", type=float, default=0.01)
    parser.add_argument("--max-n-sim-runs5-missing-rate", type=float, default=0.01)
    parser.add_argument("--max-winner-races-0", type=int, default=0)
    parser.add_argument("--max-races-lt-min-horses", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"not found: {input_path}")

    df = pd.read_parquet(input_path)
    if df.empty:
        raise SystemExit("dataset is empty")

    required_cols = set(FEATURE_COLS) | {"race_id", "race_date", "surface", "distance_m", TARGET_COL}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise SystemExit(f"missing required columns: {missing_cols}")

    monitored_feature_set = set(FEATURE_COLS)

    placeholder = (df["surface"] == 0) | (df["distance_m"] == 0)
    dist_lt800 = df["distance_m"] < 800
    dist_lt100 = (df["distance_m"] > 0) & (df["distance_m"] < 100)
    going_zero = df["going"] == 0
    field_size_zero = df["field_size"] == 0
    class_code_missing = df["class_code"].isna()

    n_runs_5_missing = (
        df["n_runs_5"].isna()
        if "n_runs_5" in monitored_feature_set and "n_runs_5" in df.columns
        else pd.Series([], dtype="bool")
    )
    n_sim_runs_5_missing = (
        df["n_sim_runs_5"].isna()
        if "n_sim_runs_5" in monitored_feature_set and "n_sim_runs_5" in df.columns
        else pd.Series([], dtype="bool")
    )
    n_runs_5_missing_rate = float(n_runs_5_missing.mean()) if len(n_runs_5_missing) else None
    n_sim_runs_5_missing_rate = float(n_sim_runs_5_missing.mean()) if len(n_sim_runs_5_missing) else None

    Xtr, _, Xes, _, Xte, _ = _split_by_race_id(
        df,
        df,
        df[TARGET_COL],
        test_size=args.test_size,
        es_val_size=args.es_val_size,
    )

    split_stats = [
        _calc_split_stats("train", Xtr, args.min_horses),
        _calc_split_stats("es", Xes, args.min_horses),
        _calc_split_stats("test", Xte, args.min_horses),
    ]

    feat = df[FEATURE_COLS]
    miss_rate = feat.isna().mean().sort_values(ascending=False)
    always_missing = miss_rate[miss_rate == 1.0].index.tolist()
    missing_gt50 = miss_rate[miss_rate > 0.5].index.tolist()
    missing_gt20 = miss_rate[miss_rate > 0.2].index.tolist()

    nunique = feat.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()

    print(f"input={input_path}")
    print(f"rows={len(df)} cols={df.shape[1]} races={df['race_id'].nunique()}")
    print(f"date_range={_fmt_date(df['race_date'].min())}..{_fmt_date(df['race_date'].max())}")
    print(f"features={len(FEATURE_COLS)} placeholder_rate={placeholder.mean():.2%}")
    anomaly_parts = [
        f"dist<800={int(dist_lt800.sum())} ({dist_lt800.mean():.2%})",
        f"dist(0<d<100)={int(dist_lt100.sum())} ({dist_lt100.mean():.4%})",
        f"going==0={int(going_zero.sum())} ({going_zero.mean():.2%})",
        f"field_size==0={int(field_size_zero.sum())} ({field_size_zero.mean():.2%})",
        f"class_code_missing={int(class_code_missing.sum())} ({class_code_missing.mean():.2%})",
    ]
    if len(n_runs_5_missing):
        anomaly_parts.append(f"n_runs_5_missing={int(n_runs_5_missing.sum())} ({n_runs_5_missing_rate:.2%})")
    if len(n_sim_runs_5_missing):
        anomaly_parts.append(
            f"n_sim_runs_5_missing={int(n_sim_runs_5_missing.sum())} ({n_sim_runs_5_missing_rate:.2%})"
        )
    print("anomalies: " + " ".join(anomaly_parts))
    for s in split_stats:
        print(
            f"{s.name}: rows={s.rows} races={s.races} pos={s.positives} ({s.pos_rate:.2%}) "
            f"dates={s.date_min}..{s.date_max} "
            f"winners_per_race 0:{s.winner_races_0} 1:{s.winner_races_1} 2:{s.winner_races_2} >2:{s.winner_races_gt2} "
            f"races<{args.min_horses}horses:{s.races_lt_min_horses} placeholder_rate:{s.placeholder_rate:.2%}"
        )
    print(f"always_missing_features={len(always_missing)}")
    if always_missing:
        print("  " + ", ".join(always_missing))
    print(f"features_missing>50%={len(missing_gt50)}")
    print(f"features_missing>20%={len(missing_gt20)}")
    print(f"constant_features(dropna=False)={len(constant)}")
    if constant:
        print("  " + ", ".join(constant))

    top_missing = miss_rate.head(10)
    print("top_missing_features (rate):")
    for col, rate in top_missing.items():
        print(f"  {col}: {rate:.2%}")

    result = {
        "input": str(input_path),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "races": int(df["race_id"].nunique()),
        "date_min": _fmt_date(df["race_date"].min()),
        "date_max": _fmt_date(df["race_date"].max()),
        "features": int(len(FEATURE_COLS)),
        "placeholder_rate": float(placeholder.mean()),
        "anomalies": {
            "dist_lt800_rate": float(dist_lt800.mean()),
            "dist_lt100_rate": float(dist_lt100.mean()),
            "going_zero_rate": float(going_zero.mean()),
            "field_size_zero_rate": float(field_size_zero.mean()),
            "class_code_missing_rate": float(class_code_missing.mean()),
        },
        "splits": [asdict(s) for s in split_stats],
        "always_missing_features": always_missing,
        "features_missing_gt50": missing_gt50,
        "features_missing_gt20": missing_gt20,
        "constant_features": constant,
    }
    if n_runs_5_missing_rate is not None:
        result["anomalies"]["n_runs_5_missing_rate"] = n_runs_5_missing_rate
    if n_sim_runs_5_missing_rate is not None:
        result["anomalies"]["n_sim_runs_5_missing_rate"] = n_sim_runs_5_missing_rate

    gate_failures: list[str] = []
    if float(placeholder.mean()) > args.max_placeholder_rate:
        gate_failures.append(
            f"placeholder_rate={placeholder.mean():.4%} > {args.max_placeholder_rate:.4%}"
        )
    if float(dist_lt800.mean()) > args.max_dist_lt800_rate:
        gate_failures.append(f"dist_lt800_rate={dist_lt800.mean():.4%} > {args.max_dist_lt800_rate:.4%}")
    if float(dist_lt100.mean()) > args.max_dist_lt100_rate:
        gate_failures.append(f"dist_lt100_rate={dist_lt100.mean():.4%} > {args.max_dist_lt100_rate:.4%}")
    if float(going_zero.mean()) > args.max_going_zero_rate:
        gate_failures.append(f"going_zero_rate={going_zero.mean():.4%} > {args.max_going_zero_rate:.4%}")
    if float(field_size_zero.mean()) > args.max_field_size_zero_rate:
        gate_failures.append(
            f"field_size_zero_rate={field_size_zero.mean():.4%} > {args.max_field_size_zero_rate:.4%}"
        )
    if float(class_code_missing.mean()) > args.max_class_code_missing_rate:
        gate_failures.append(
            f"class_code_missing_rate={class_code_missing.mean():.4%} > {args.max_class_code_missing_rate:.4%}"
        )
    if len(n_runs_5_missing) and n_runs_5_missing_rate is not None and n_runs_5_missing_rate > args.max_n_runs5_missing_rate:
        gate_failures.append(
            f"n_runs_5_missing_rate={n_runs_5_missing_rate:.4%} > {args.max_n_runs5_missing_rate:.4%}"
        )
    if (
        len(n_sim_runs_5_missing)
        and n_sim_runs_5_missing_rate is not None
        and n_sim_runs_5_missing_rate > args.max_n_sim_runs5_missing_rate
    ):
        gate_failures.append(
            "n_sim_runs_5_missing_rate="
            f"{n_sim_runs_5_missing_rate:.4%} > {args.max_n_sim_runs5_missing_rate:.4%}"
        )
    if "distance_change_m" in constant:
        gate_failures.append("distance_change_m is constant")

    for split in split_stats:
        if split.winner_races_0 > args.max_winner_races_0:
            gate_failures.append(
                f"{split.name}.winner_races_0={split.winner_races_0} > {args.max_winner_races_0}"
            )
        if split.races_lt_min_horses > args.max_races_lt_min_horses:
            gate_failures.append(
                f"{split.name}.races_lt_min_horses={split.races_lt_min_horses} > {args.max_races_lt_min_horses}"
            )

    result["gate"] = {"enabled": bool(args.gate), "passed": len(gate_failures) == 0, "failures": gate_failures}

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    if args.gate:
        if gate_failures:
            print("QUALITY_GATE: FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
            raise SystemExit(1)
        print("QUALITY_GATE: PASS")


if __name__ == "__main__":
    main()
