from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PLSamplingConfig:
    mc_samples: int = 10_000
    top_k: int = 3
    eps: float = 1e-6
    seed: int = 42


def make_race_rng(seed: int, race_id: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(seed), int(race_id)])
    return np.random.default_rng(seed_seq)


def kumiban_from_horse_nos(horse_no_1: int, horse_no_2: int) -> str:
    left, right = sorted((int(horse_no_1), int(horse_no_2)))
    return f"{left:02d}{right:02d}"


def _pl_weights_from_p_top3(p_top3: np.ndarray, *, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p_top3, dtype=float), float(eps), 1.0 - float(eps))
    weights = p / (1.0 - p)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        return np.ones_like(p, dtype=float)
    return np.maximum(weights, float(eps))


def estimate_wide_probabilities_for_race(
    race_df: pd.DataFrame,
    *,
    mc_samples: int,
    top_k: int,
    eps: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    required = {"race_id", "horse_no", "p_top3"}
    missing = sorted(required - set(race_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in race_df: {missing}")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be > 0")
    if top_k <= 1:
        raise ValueError("top_k must be > 1")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    work = race_df.copy()
    work["horse_no"] = pd.to_numeric(work["horse_no"], errors="coerce").astype("Int64")
    work["p_top3"] = pd.to_numeric(work["p_top3"], errors="coerce")
    work = work.dropna(subset=["horse_no", "p_top3"]).copy()
    work["horse_no"] = work["horse_no"].astype(int)
    work = work.sort_values("horse_no", kind="mergesort").reset_index(drop=True)

    n_horses = len(work)
    if n_horses < 2:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no_1",
                "horse_no_2",
                "kumiban",
                "p_wide",
                "p_top3_1",
                "p_top3_2",
            ]
        )

    sample_k = min(int(top_k), int(n_horses))
    race_id = int(work["race_id"].iloc[0])
    horse_nos = work["horse_no"].to_numpy(dtype=int)
    p_top3 = work["p_top3"].to_numpy(dtype=float)

    weights = _pl_weights_from_p_top3(p_top3, eps=eps)
    log_weights = np.log(weights)

    # Gumbel top-k trick samples PL-without-replacement efficiently.
    scores = rng.gumbel(size=(int(mc_samples), int(n_horses))) + log_weights[np.newaxis, :]
    topk_indices = np.argpartition(scores, -sample_k, axis=1)[:, -sample_k:]

    # NOTE: Use a sufficiently wide integer dtype. uint8 overflows at 255 and
    # breaks co-occurrence counts when mc_samples is large (e.g., 10_000).
    selected_mask = np.zeros((int(mc_samples), int(n_horses)), dtype=np.int32)
    row_indices = np.arange(int(mc_samples), dtype=np.int32)[:, np.newaxis]
    selected_mask[row_indices, topk_indices] = 1

    co_counts = selected_mask.T @ selected_mask
    co_probs = co_counts.astype(float) / float(mc_samples)

    rows: list[dict[str, float | int | str]] = []
    for left in range(n_horses - 1):
        for right in range(left + 1, n_horses):
            horse_no_1 = int(horse_nos[left])
            horse_no_2 = int(horse_nos[right])
            rows.append(
                {
                    "race_id": race_id,
                    "horse_no_1": horse_no_1,
                    "horse_no_2": horse_no_2,
                    "kumiban": kumiban_from_horse_nos(horse_no_1, horse_no_2),
                    "p_wide": float(co_probs[left, right]),
                    "p_top3_1": float(p_top3[left]),
                    "p_top3_2": float(p_top3[right]),
                }
            )

    return pd.DataFrame(rows)
