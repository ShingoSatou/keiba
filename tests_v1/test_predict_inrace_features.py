from decimal import Decimal

import pandas as pd

from scripts.predict import _add_inrace_features


def test_add_inrace_features_handles_decimal():
    frame = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "speed_best_5": [Decimal("10.5"), Decimal("11.5"), Decimal("9.5")],
            "closing_best_5": [Decimal("1.2"), Decimal("1.0"), Decimal("1.1")],
            "early_mean_3": [Decimal("3.3"), Decimal("3.1"), Decimal("3.2")],
        }
    )
    out = _add_inrace_features(frame)
    for col in [
        "speed_best_5_z_inrace",
        "speed_best_5_rank",
        "closing_best_5_z_inrace",
        "closing_best_5_rank",
        "early_mean_3_z_inrace",
        "early_mean_3_rank",
    ]:
        assert col in out.columns
        assert out[col].isna().sum() == 0
