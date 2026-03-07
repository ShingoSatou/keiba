from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts_v3.metrics_benter_v3_common import logit_clip

logger = logging.getLogger(__name__)


def ensure_torch_available():
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return None
    return torch


@dataclass(frozen=True)
class PLTrainConfig:
    epochs: int = 300
    lr: float = 0.05
    l2: float = 1e-5
    seed: int = 42


@dataclass(frozen=True)
class PLSamplingConfig:
    mc_samples: int = 10000
    top_k: int = 3
    seed: int = 42


def _race_centered(series: pd.Series, race_id: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric - numeric.groupby(race_id, sort=False).transform("mean")


def _race_rank_pct(series: pd.Series, race_id: pd.Series, *, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.groupby(race_id, sort=False).rank(
        method="average",
        pct=True,
        ascending=ascending,
    )


def materialize_stack_default_pl_features(
    frame: pd.DataFrame,
    *,
    eps: float = 1e-6,
) -> pd.DataFrame:
    required = {
        "race_id",
        "p_win_stack",
        "p_place_stack",
        "place_width_log_ratio",
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for stack_default PL features: {missing}")

    out = frame.copy()
    out["z_win_stack"] = logit_clip(pd.to_numeric(out["p_win_stack"], errors="coerce"), eps=eps)
    out["z_place_stack"] = logit_clip(
        pd.to_numeric(out["p_place_stack"], errors="coerce"),
        eps=eps,
    )
    out["place_width_log_ratio"] = pd.to_numeric(out["place_width_log_ratio"], errors="coerce")

    out["z_win_stack_x_z_place_stack"] = out["z_win_stack"] * out["z_place_stack"]
    out["z_win_stack_x_place_width_log_ratio"] = (
        out["z_win_stack"] * out["place_width_log_ratio"]
    )
    out["z_place_stack_x_place_width_log_ratio"] = (
        out["z_place_stack"] * out["place_width_log_ratio"]
    )
    out["z_win_stack_x_field_size"] = out["z_win_stack"] * pd.to_numeric(
        out["field_size"],
        errors="coerce",
    )
    out["z_place_stack_x_field_size"] = out["z_place_stack"] * pd.to_numeric(
        out["field_size"],
        errors="coerce",
    )
    out["z_win_stack_x_distance_m"] = out["z_win_stack"] * pd.to_numeric(
        out["distance_m"],
        errors="coerce",
    )
    out["z_place_stack_x_distance_m"] = out["z_place_stack"] * pd.to_numeric(
        out["distance_m"],
        errors="coerce",
    )

    race_id = out["race_id"]
    out["z_win_stack_race_centered"] = _race_centered(out["z_win_stack"], race_id)
    out["z_place_stack_race_centered"] = _race_centered(out["z_place_stack"], race_id)
    out["place_width_log_ratio_race_centered"] = _race_centered(
        out["place_width_log_ratio"],
        race_id,
    )
    out["z_win_stack_rank_pct"] = _race_rank_pct(out["z_win_stack"], race_id, ascending=False)
    out["z_place_stack_rank_pct"] = _race_rank_pct(
        out["z_place_stack"],
        race_id,
        ascending=False,
    )
    out["place_width_log_ratio_rank_pct"] = _race_rank_pct(
        out["place_width_log_ratio"],
        race_id,
        ascending=True,
    )
    return out


def make_race_rng(seed: int, race_id: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(seed), int(race_id)])
    return np.random.default_rng(seed_seq)


def build_group_indices(
    frame: pd.DataFrame,
    *,
    race_col: str = "race_id",
    finish_col: str = "finish_pos",
    horse_no_col: str = "horse_no",
) -> list[np.ndarray]:
    work = frame.copy()
    work[finish_col] = pd.to_numeric(work[finish_col], errors="coerce")
    work[horse_no_col] = pd.to_numeric(work[horse_no_col], errors="coerce")
    work = work[work[finish_col].notna() & work[horse_no_col].notna()].copy()
    work = work.sort_values([race_col, finish_col, horse_no_col], kind="mergesort")

    groups: list[np.ndarray] = []
    for _, sub in work.groupby(race_col, sort=False):
        if len(sub) < 2:
            continue
        groups.append(sub.index.to_numpy(dtype=np.int64))
    return groups


def pl_nll_numpy(scores: np.ndarray, groups: list[np.ndarray]) -> float:
    x = np.asarray(scores, dtype=float)
    total = 0.0
    count = 0
    for idx in groups:
        s = x[idx]
        if s.size < 2 or not np.all(np.isfinite(s)):
            continue
        for pos in range(s.size):
            block = s[pos:]
            max_block = float(np.max(block))
            log_denom = float(max_block + np.log(np.sum(np.exp(block - max_block))))
            total += log_denom - float(s[pos])
        count += 1
    if count == 0:
        return float("nan")
    return float(total / count)


def fit_pl_linear_torch(
    x_train: np.ndarray,
    groups: list[np.ndarray],
    *,
    config: PLTrainConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    torch = ensure_torch_available()
    if torch is None:  # pragma: no cover
        logger.warning("torch is not installed. Falling back to numpy PL optimizer.")
        return _fit_pl_linear_numpy(x_train, groups, config=config)

    if x_train.ndim != 2:
        raise ValueError("x_train must be 2-D")
    if len(groups) == 0:
        raise ValueError("No valid race groups to train PL")

    torch.manual_seed(int(config.seed))
    x_t = torch.tensor(x_train, dtype=torch.float32)
    group_tensors = [torch.tensor(g, dtype=torch.long) for g in groups]

    w = torch.nn.Parameter(torch.zeros(x_t.shape[1], dtype=torch.float32))
    optimizer = torch.optim.Adam([w], lr=float(config.lr), weight_decay=float(config.l2))

    final_loss = float("nan")
    for _ in range(int(config.epochs)):
        optimizer.zero_grad()
        scores = x_t @ w

        total = torch.tensor(0.0, dtype=torch.float32)
        valid_groups = 0
        for idx in group_tensors:
            s = scores[idx]
            if s.numel() < 2:
                continue
            race_loss = torch.tensor(0.0, dtype=torch.float32)
            for pos in range(int(s.numel())):
                remain = s[pos:]
                race_loss = race_loss + torch.logsumexp(remain, dim=0) - s[pos]
            total = total + race_loss
            valid_groups += 1

        if valid_groups <= 0:
            raise ValueError("No valid race groups found during optimization")
        loss = total / float(valid_groups)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    w_np = w.detach().cpu().numpy().astype(float)
    return w_np, {"train_nll": float(final_loss)}


def _pl_nll_and_grad_numpy(
    x_train: np.ndarray,
    groups: list[np.ndarray],
    w: np.ndarray,
) -> tuple[float, np.ndarray]:
    scores = x_train @ w
    grad = np.zeros_like(w, dtype=float)
    total_loss = 0.0
    valid_groups = 0

    for idx in groups:
        s = scores[idx]
        xg = x_train[idx]
        if s.size < 2 or not np.all(np.isfinite(s)):
            continue

        race_loss = 0.0
        race_grad = np.zeros_like(w, dtype=float)
        for pos in range(s.size):
            rem_scores = s[pos:]
            rem_x = xg[pos:, :]
            max_block = float(np.max(rem_scores))
            exp_block = np.exp(rem_scores - max_block)
            denom = float(np.sum(exp_block))
            probs = exp_block / denom

            race_loss += float(max_block + np.log(denom) - rem_scores[0])

            grad_scores = probs
            grad_scores[0] -= 1.0
            race_grad += rem_x.T @ grad_scores

        total_loss += race_loss
        grad += race_grad
        valid_groups += 1

    if valid_groups <= 0:
        raise ValueError("No valid race groups found during optimization")
    return float(total_loss / valid_groups), grad / float(valid_groups)


def _fit_pl_linear_numpy(
    x_train: np.ndarray,
    groups: list[np.ndarray],
    *,
    config: PLTrainConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2-D")
    if len(groups) == 0:
        raise ValueError("No valid race groups to train PL")

    rng = np.random.default_rng(int(config.seed))
    w = rng.normal(loc=0.0, scale=1e-3, size=x_train.shape[1]).astype(float)

    lr = float(config.lr)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    final_loss = float("nan")

    for step in range(1, int(config.epochs) + 1):
        loss, grad = _pl_nll_and_grad_numpy(x_train, groups, w)
        grad = grad + float(config.l2) * w

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        final_loss = float(loss)

    return w.astype(float), {"train_nll": float(final_loss)}


def predict_linear_scores(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) @ np.asarray(w, dtype=float)


def estimate_topk_probs_from_scores(
    scores: np.ndarray,
    *,
    top_k: int,
    mc_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    n = s.shape[0]
    if n == 0:
        return np.array([], dtype=float), np.zeros((0, 0), dtype=float)
    sample_k = min(max(1, int(top_k)), int(n))
    if int(mc_samples) <= 0:
        raise ValueError("mc_samples must be > 0")

    sampled = rng.gumbel(size=(int(mc_samples), int(n))) + s[np.newaxis, :]
    top_idx = np.argpartition(sampled, -sample_k, axis=1)[:, -sample_k:]

    selected = np.zeros((int(mc_samples), int(n)), dtype=np.int32)
    row_idx = np.arange(int(mc_samples), dtype=np.int32)[:, np.newaxis]
    selected[row_idx, top_idx] = 1

    p_topk = selected.mean(axis=0).astype(float)
    co_prob = (selected.T @ selected).astype(float) / float(mc_samples)
    return p_topk, co_prob


def estimate_p_top3_by_race(
    frame: pd.DataFrame,
    *,
    score_col: str,
    mc_samples: int,
    seed: int,
    top_k: int = 3,
) -> pd.DataFrame:
    required = {"race_id", "horse_no", score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    outputs: list[pd.DataFrame] = []
    for race_id, sub in frame.groupby("race_id", sort=False):
        race = sub.sort_values("horse_no", kind="mergesort").copy()
        scores = pd.to_numeric(race[score_col], errors="coerce").to_numpy(dtype=float)
        rng = make_race_rng(seed, int(race_id))
        p_top3, _ = estimate_topk_probs_from_scores(
            scores,
            top_k=top_k,
            mc_samples=mc_samples,
            rng=rng,
        )
        race["p_top3"] = p_top3
        outputs.append(race[["race_id", "horse_no", "p_top3"]])

    if not outputs:
        return pd.DataFrame(columns=["race_id", "horse_no", "p_top3"])
    out = pd.concat(outputs, axis=0, ignore_index=True)
    return out.sort_values(["race_id", "horse_no"], kind="mergesort")


def estimate_p_wide_by_race(
    frame: pd.DataFrame,
    *,
    score_col: str,
    mc_samples: int,
    seed: int,
    top_k: int = 3,
) -> pd.DataFrame:
    required = {"race_id", "horse_no", score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, float | int | str]] = []
    for race_id, sub in frame.groupby("race_id", sort=False):
        race = sub.sort_values("horse_no", kind="mergesort").copy()
        horse_nos = pd.to_numeric(race["horse_no"], errors="coerce").to_numpy(dtype=int)
        scores = pd.to_numeric(race[score_col], errors="coerce").to_numpy(dtype=float)
        rng = make_race_rng(seed, int(race_id))
        p_top3, co_prob = estimate_topk_probs_from_scores(
            scores,
            top_k=top_k,
            mc_samples=mc_samples,
            rng=rng,
        )
        n = len(horse_nos)
        for i in range(n - 1):
            for j in range(i + 1, n):
                left = int(min(horse_nos[i], horse_nos[j]))
                right = int(max(horse_nos[i], horse_nos[j]))
                rows.append(
                    {
                        "race_id": int(race_id),
                        "horse_no_1": left,
                        "horse_no_2": right,
                        "kumiban": f"{left:02d}{right:02d}",
                        "p_wide": float(co_prob[i, j]),
                        "p_top3_1": float(p_top3[i]),
                        "p_top3_2": float(p_top3[j]),
                    }
                )

    return pd.DataFrame(rows)
