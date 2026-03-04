from __future__ import annotations

import math

import numpy as np


def logit_clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities then return logit scores."""
    if not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must be in (0, 0.5)")
    p_arr = np.asarray(p, dtype=float)
    clipped = np.clip(p_arr, float(eps), 1.0 - float(eps))
    return np.log(clipped / (1.0 - clipped))


def race_softmax(scores: np.ndarray, race_id: np.ndarray, beta: float) -> np.ndarray:
    """Race-wise softmax with numerical stability (logsumexp style)."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(race_id)
    if s.shape[0] != r.shape[0]:
        raise ValueError("scores and race_id length mismatch")
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be > 0 and finite")

    out = np.full_like(s, np.nan, dtype=float)
    valid = np.isfinite(s)
    if not valid.any():
        return out

    idx = np.where(valid)[0]
    r_valid = r[idx]
    s_valid = s[idx]

    order = np.argsort(r_valid, kind="mergesort")
    idx_ord = idx[order]
    r_ord = r_valid[order]
    s_ord = s_valid[order]

    start = 0
    n = len(idx_ord)
    while start < n:
        end = start + 1
        while end < n and r_ord[end] == r_ord[start]:
            end += 1

        block = float(beta) * s_ord[start:end]
        max_block = float(np.max(block))
        exp_block = np.exp(block - max_block)
        denom = float(np.sum(exp_block))
        if np.isfinite(denom) and denom > 0.0:
            out[idx_ord[start:end]] = exp_block / denom

        start = end

    return out


def benter_nll_and_null(
    race_id: np.ndarray,
    y_win: np.ndarray,
    field_size: np.ndarray,
    c_prob: np.ndarray,
) -> tuple[float, float, int]:
    """Compute NLL(model), NLL(null), valid-race count for Benter R*."""
    r = np.asarray(race_id)
    y = np.asarray(y_win, dtype=float)
    f = np.asarray(field_size, dtype=float)
    c = np.asarray(c_prob, dtype=float)

    n = r.shape[0]
    if y.shape[0] != n or f.shape[0] != n or c.shape[0] != n:
        raise ValueError("Input length mismatch")

    order = np.argsort(r, kind="mergesort")
    r_ord = r[order]
    y_ord = y[order]
    f_ord = f[order]
    c_ord = c[order]

    nll_model = 0.0
    nll_null = 0.0
    n_races = 0

    start = 0
    while start < n:
        end = start + 1
        while end < n and r_ord[end] == r_ord[start]:
            end += 1

        y_block = y_ord[start:end]
        winner_idx = np.where(y_block == 1.0)[0]
        if winner_idx.size != 1:
            start = end
            continue

        c_winner = float(c_ord[start:end][winner_idx[0]])
        if not np.isfinite(c_winner) or c_winner <= 0.0:
            start = end
            continue

        f_block = f_ord[start:end]
        f_valid = f_block[np.isfinite(f_block)]
        if f_valid.size == 0:
            start = end
            continue
        race_field_size = float(np.max(f_valid))
        if race_field_size <= 1.0:
            start = end
            continue

        nll_model -= float(np.log(np.clip(c_winner, 1e-12, 1.0)))
        nll_null += float(np.log(race_field_size))
        n_races += 1
        start = end

    return float(nll_model), float(nll_null), int(n_races)


def _nll_objective(
    beta: float,
    *,
    race_id: np.ndarray,
    y_win: np.ndarray,
    field_size: np.ndarray,
    scores: np.ndarray,
) -> float:
    c_prob = race_softmax(scores, race_id, beta=float(beta))
    nll_model, _, n_races = benter_nll_and_null(race_id, y_win, field_size, c_prob)
    if n_races <= 0 or not np.isfinite(nll_model):
        return float("inf")
    return float(nll_model)


def fit_beta_by_nll(
    race_id: np.ndarray,
    y_win: np.ndarray,
    field_size: np.ndarray,
    scores_train: np.ndarray,
    beta_min: float = 0.01,
    beta_max: float = 100.0,
) -> float:
    """Fit beta by minimizing train NLL without SciPy (grid + golden search)."""
    if not (0.0 < float(beta_min) < float(beta_max)):
        raise ValueError("Require 0 < beta_min < beta_max")

    low = float(beta_min)
    high = float(beta_max)

    grid = np.logspace(np.log10(low), np.log10(high), 81)
    obj = np.array(
        [
            _nll_objective(
                float(b),
                race_id=race_id,
                y_win=y_win,
                field_size=field_size,
                scores=scores_train,
            )
            for b in grid
        ],
        dtype=float,
    )

    finite = np.isfinite(obj)
    if not finite.any():
        return 1.0

    best_idx = int(np.nanargmin(obj))
    best_beta = float(grid[best_idx])

    left_idx = max(0, best_idx - 1)
    right_idx = min(len(grid) - 1, best_idx + 1)
    left = float(grid[left_idx])
    right = float(grid[right_idx])
    if left == right:
        return float(np.clip(best_beta, low, high))

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - phi * (right - left)
    d = left + phi * (right - left)
    fc = _nll_objective(
        c,
        race_id=race_id,
        y_win=y_win,
        field_size=field_size,
        scores=scores_train,
    )
    fd = _nll_objective(
        d,
        race_id=race_id,
        y_win=y_win,
        field_size=field_size,
        scores=scores_train,
    )

    for _ in range(48):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - phi * (right - left)
            fc = _nll_objective(
                c,
                race_id=race_id,
                y_win=y_win,
                field_size=field_size,
                scores=scores_train,
            )
        else:
            left = c
            c = d
            fc = fd
            d = left + phi * (right - left)
            fd = _nll_objective(
                d,
                race_id=race_id,
                y_win=y_win,
                field_size=field_size,
                scores=scores_train,
            )

    if fc <= fd:
        best_beta = float(c)
        best_obj = float(fc)
    else:
        best_beta = float(d)
        best_obj = float(fd)

    if not np.isfinite(best_obj):
        return float(np.clip(grid[best_idx], low, high))
    return float(np.clip(best_beta, low, high))


def benter_r2(nll_model: float, nll_null: float) -> float:
    """Pseudo R* defined as 1 - NLL_model / NLL_null."""
    if not np.isfinite(nll_model) or not np.isfinite(nll_null) or float(nll_null) <= 0.0:
        return float("nan")
    return float(1.0 - (float(nll_model) / float(nll_null)))
