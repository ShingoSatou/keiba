from __future__ import annotations

import pickle
import sys

from scripts.predict import _load_pickle_with_compat


def test_load_pickle_with_main_wrapper_compat(tmp_path):
    main_mod = sys.modules["__main__"]
    original = getattr(main_mod, "LGBMClassifierWrapper", None)
    had_original = hasattr(main_mod, "LGBMClassifierWrapper")

    class _TempWrapper:
        def __init__(self):
            self.model = None
            self.feature_names = ["f1"]
            self.params = {}
            self.num_boost_round = 123

    _TempWrapper.__module__ = "__main__"
    _TempWrapper.__name__ = "LGBMClassifierWrapper"
    _TempWrapper.__qualname__ = "LGBMClassifierWrapper"
    main_mod.LGBMClassifierWrapper = _TempWrapper

    payload_path = tmp_path / "model.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump({"model": _TempWrapper(), "feature_names": ["f1"]}, f)

    delattr(main_mod, "LGBMClassifierWrapper")
    loaded = _load_pickle_with_compat(payload_path)

    if had_original:
        main_mod.LGBMClassifierWrapper = original

    assert loaded["model"].__class__.__name__ == "LGBMClassifierWrapper"
    assert loaded["feature_names"] == ["f1"]
