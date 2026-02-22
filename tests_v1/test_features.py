"""
特徴量計算ロジックのテスト

build_features.py / build_dataset.py の計算ロジックをテストする。
"""

import math

# =============================================================================
# distance_to_bucket のテスト
# =============================================================================


def distance_to_bucket(distance_m: int) -> int:
    """距離をバケットに変換 (scripts/build_features.py からコピー)"""
    buckets = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3200, 3600]
    for bucket in buckets:
        if distance_m <= bucket:
            return bucket
    return buckets[-1]


class TestDistanceToBucket:
    """距離バケット変換のテスト"""

    def test_exact_match(self):
        """ちょうどバケット境界の場合"""
        assert distance_to_bucket(1200) == 1200
        assert distance_to_bucket(1600) == 1600
        assert distance_to_bucket(2000) == 2000

    def test_below_bucket(self):
        """バケット未満の場合: そのバケットに入る"""
        assert distance_to_bucket(1150) == 1200  # 1000 < 1150 <= 1200
        assert distance_to_bucket(1550) == 1600  # 1400 < 1550 <= 1600
        assert distance_to_bucket(1900) == 2000  # 1800 < 1900 <= 2000

    def test_minimum_distance(self):
        """最小距離 (芝1000m)"""
        assert distance_to_bucket(1000) == 1000
        assert distance_to_bucket(800) == 1000  # 1000未満も1000に

    def test_maximum_distance(self):
        """最大距離 (長距離)"""
        assert distance_to_bucket(3600) == 3600
        assert distance_to_bucket(4000) == 3600  # オーバーも最大バケットに

    def test_common_distances(self):
        """よく使われる距離"""
        assert distance_to_bucket(1200) == 1200  # スプリント
        assert distance_to_bucket(1600) == 1600  # マイル
        assert distance_to_bucket(2000) == 2000  # 中距離
        assert distance_to_bucket(2400) == 2400  # クラシック
        assert distance_to_bucket(3200) == 3200  # 長距離


# =============================================================================
# going_to_bucket のテスト
# =============================================================================


def going_to_bucket(going: int | None) -> int:
    """馬場状態をバケットに変換 (scripts/build_features.py からコピー)"""
    if going is None or math.isnan(going) or going <= 2:
        return 1  # 良系 (良, 稍重)
    return 2  # 道悪系 (重, 不良)


class TestGoingToBucket:
    """馬場バケット変換のテスト"""

    def test_good_going(self):
        """良・稍重は良系 (1)"""
        assert going_to_bucket(1) == 1  # 良
        assert going_to_bucket(2) == 1  # 稍重

    def test_bad_going(self):
        """重・不良は道悪系 (2)"""
        assert going_to_bucket(3) == 2  # 重
        assert going_to_bucket(4) == 2  # 不良

    def test_none_going(self):
        """欠損は良系扱い"""
        assert going_to_bucket(None) == 1

    def test_nan_going(self):
        """NaN も欠損として良系扱い"""
        assert going_to_bucket(math.nan) == 1


# =============================================================================
# is_similar_condition のテスト
# =============================================================================


def is_similar_condition(
    target_surface: int,
    target_distance_bucket: int,
    target_going_bucket: int,
    past_surface: int,
    past_distance_m: int,
    past_going: int | None,
) -> bool:
    """条件近似かどうかを判定 (scripts/build_features.py からコピー)"""
    # 1) surface: 完全一致必須
    if past_surface != target_surface:
        return False

    # 2) distance: ±200m 以内
    if abs(past_distance_m - target_distance_bucket) > 200:
        return False

    # 3) going: 良系/道悪系 で一致
    if going_to_bucket(past_going) != target_going_bucket:
        return False

    return True


class TestIsSimilarCondition:
    """条件近似判定のテスト"""

    def test_exact_match(self):
        """完全一致"""
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=1600,
            past_going=1,
        )

    def test_distance_within_200m(self):
        """距離±200m以内はマッチ"""
        # 1600m ターゲットに対して 1400m (差200m) はOK
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=1400,
            past_going=1,
        )

        # 1800m (差200m) もOK
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=1800,
            past_going=1,
        )

    def test_distance_over_200m(self):
        """距離±200m超はマッチしない"""
        # 1600m ターゲットに対して 1200m (差400m) はNG
        assert not is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=1200,
            past_going=1,
        )

    def test_surface_mismatch(self):
        """surface 不一致はマッチしない"""
        # 芝 vs ダート
        assert not is_similar_condition(
            target_surface=1,  # 芝
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=2,  # ダート
            past_distance_m=1600,
            past_going=1,
        )

    def test_going_mismatch(self):
        """going 系統不一致はマッチしない"""
        # 良系 vs 道悪系
        assert not is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,  # 良系
            past_surface=1,
            past_distance_m=1600,
            past_going=3,  # 重 (道悪系)
        )

    def test_going_same_group(self):
        """going 同系統はマッチ"""
        # 良 (1) と 稍重 (2) は同じ良系
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,  # 良系
            past_surface=1,
            past_distance_m=1600,
            past_going=2,  # 稍重 (良系)
        )

        # 重 (3) と 不良 (4) は同じ道悪系
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=2,  # 道悪系
            past_surface=1,
            past_distance_m=1600,
            past_going=4,  # 不良 (道悪系)
        )

    def test_real_scenario_turf_mile(self):
        """実際のシナリオ: 芝マイル良"""
        # 東京芝1600m良 vs 中山芝1800m良 → OK
        assert is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=1800,
            past_going=1,
        )

        # 東京芝1600m良 vs 東京ダート1600m良 → NG (surface違い)
        assert not is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=2,
            past_distance_m=1600,
            past_going=1,
        )

        # 東京芝1600m良 vs 京都芝2000m良 → NG (距離差400m)
        assert not is_similar_condition(
            target_surface=1,
            target_distance_bucket=1600,
            target_going_bucket=1,
            past_surface=1,
            past_distance_m=2000,
            past_going=1,
        )
