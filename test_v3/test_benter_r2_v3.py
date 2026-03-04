from __future__ import annotations

import numpy as np
import pytest

from scripts_v3.metrics_benter_v3_common import (
    benter_nll_and_null,
    benter_r2,
    fit_beta_by_nll,
    logit_clip,
    race_softmax,
)


def test_race_softmax_sums_to_one_per_race() -> None:
    race_id = np.array([1, 1, 2, 2, 2], dtype=int)
    p = np.array([0.8, 0.2, 0.6, 0.3, 0.1], dtype=float)
    scores = logit_clip(p, eps=1e-6)
    c = race_softmax(scores, race_id, beta=1.0)

    assert np.isfinite(c).all()
    assert c[race_id == 1].sum() == pytest.approx(1.0)
    assert c[race_id == 2].sum() == pytest.approx(1.0)


def test_beta_changes_nll_and_has_better_point_than_beta1() -> None:
    race_id = np.array([1, 1, 2, 2, 3, 3], dtype=int)
    y_win = np.array([1, 0, 0, 1, 1, 0], dtype=int)
    field_size = np.array([2, 2, 2, 2, 2, 2], dtype=float)
    p = np.array([0.6, 0.4, 0.4, 0.6, 0.52, 0.48], dtype=float)
    scores = logit_clip(p, eps=1e-6)

    c_beta1 = race_softmax(scores, race_id, beta=1.0)
    nll_beta1, nll_null_beta1, _ = benter_nll_and_null(race_id, y_win, field_size, c_beta1)

    beta_hat = fit_beta_by_nll(
        race_id,
        y_win,
        field_size,
        scores,
        beta_min=0.01,
        beta_max=100.0,
    )
    assert 0.01 <= beta_hat <= 100.0

    c_betahat = race_softmax(scores, race_id, beta=beta_hat)
    nll_betahat, nll_null_betahat, _ = benter_nll_and_null(race_id, y_win, field_size, c_betahat)
    assert nll_null_betahat == pytest.approx(nll_null_beta1)
    assert nll_betahat <= nll_beta1 + 1e-10
    assert np.isfinite(benter_r2(nll_betahat, nll_null_betahat))


def test_nll_null_matches_sum_log_field_size() -> None:
    race_id = np.array([10, 10, 10, 11, 11], dtype=int)
    y_win = np.array([0, 1, 0, 1, 0], dtype=int)
    field_size = np.array([3, 3, 3, 2, 2], dtype=float)
    c = np.array([0.2, 0.6, 0.2, 0.7, 0.3], dtype=float)

    _, nll_null, n_races = benter_nll_and_null(race_id, y_win, field_size, c)
    assert n_races == 2
    assert nll_null == pytest.approx(np.log(3.0) + np.log(2.0))


def test_invalid_winner_races_are_excluded() -> None:
    race_id = np.array([1, 1, 2, 2, 3, 3], dtype=int)
    y_win = np.array(
        [
            1,
            0,  # race 1: valid
            0,
            0,  # race 2: no winner -> invalid
            1,
            1,  # race 3: multiple winners -> invalid
        ],
        dtype=int,
    )
    field_size = np.array([2, 2, 2, 2, 2, 2], dtype=float)
    c = np.array([0.7, 0.3, 0.5, 0.5, 0.6, 0.4], dtype=float)

    nll_model, nll_null, n_races = benter_nll_and_null(race_id, y_win, field_size, c)
    assert n_races == 1
    assert nll_model == pytest.approx(-np.log(0.7))
    assert nll_null == pytest.approx(np.log(2.0))
