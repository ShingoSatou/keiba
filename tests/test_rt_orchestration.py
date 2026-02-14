"""
RT オーケストレーション (backfill / ops) のテスト

DB接続を必要としないロジック部分のユニットテスト。
"""

from scripts.rt_common import (
    OPS_DEFAULT_DATASPECS,
    build_output_path,
    find_existing_output,
    group_racekeys_by_date,
    load_progress,
    race_id_to_racekey,
    save_progress,
)


class TestRaceIdToRacekey:
    """race_id → racekey 変換のテスト"""

    def test_basic_conversion(self):
        # race_id = 202602030501 → YYYYMMDD=20260203, JJ=05, RR=01
        assert race_id_to_racekey(202602030501) == "202602030501"

    def test_tokyo_race12(self):
        # 東京 (05) 12R
        assert race_id_to_racekey(202602030512) == "202602030512"

    def test_hanshin_race1(self):
        # 阪神 (09) 1R
        assert race_id_to_racekey(202602030901) == "202602030901"

    def test_small_track_code(self):
        # 札幌 (01) 5R
        assert race_id_to_racekey(202608010105) == "202608010105"

    def test_zero_padding(self):
        # race_no=1 → "01" に0パディングされるか
        result = race_id_to_racekey(202601010101)
        assert len(result) == 12
        assert result == "202601010101"

    def test_format_12chars(self):
        # 結果が常に12文字であることを確認
        result = race_id_to_racekey(202602030501)
        assert len(result) == 12


class TestBuildOutputPath:
    """出力パス生成のテスト"""

    def test_basic(self, tmp_path):
        result = build_output_path(tmp_path, "0B41", "202602030501", "20260214_120000")
        assert result == tmp_path / "0B41_202602030501_20260214_120000.jsonl"

    def test_auto_timestamp(self, tmp_path):
        result = build_output_path(tmp_path, "0B41", "202602030501")
        # ファイル名に0B41とracekeyが含まれること
        assert "0B41_202602030501_" in result.name
        assert result.suffix == ".jsonl"


class TestFindExistingOutput:
    """既存ファイル検索のテスト"""

    def test_no_existing(self, tmp_path):
        assert find_existing_output(tmp_path, "0B41", "202602030501") is None

    def test_find_existing(self, tmp_path):
        # テスト用ファイルを作成
        target = tmp_path / "0B41_202602030501_20260214_120000.jsonl"
        target.touch()
        result = find_existing_output(tmp_path, "0B41", "202602030501")
        assert result is not None
        assert result.name == target.name

    def test_different_racekey_not_matched(self, tmp_path):
        # 別の racekey のファイルはマッチしない
        (tmp_path / "0B41_202602030502_20260214_120000.jsonl").touch()
        assert find_existing_output(tmp_path, "0B41", "202602030501") is None


class TestProgressSaveLoad:
    """進捗ファイルの読み書きテスト"""

    def test_save_and_load(self, tmp_path):
        progress_file = tmp_path / "progress.json"
        data = {"completed": ["202602030501", "202602030502"]}

        save_progress(progress_file, data)
        loaded = load_progress(progress_file)

        assert loaded == data

    def test_load_nonexistent(self, tmp_path):
        progress_file = tmp_path / "nonexistent.json"
        result = load_progress(progress_file)
        assert result == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        progress_file = tmp_path / "sub" / "dir" / "progress.json"
        save_progress(progress_file, {"test": True})
        assert progress_file.exists()

    def test_roundtrip_completed_set(self, tmp_path):
        """進捗データの完全な往復テスト"""
        progress_file = tmp_path / "progress.json"
        keys = ["202602030501", "202602030502", "202602030901"]
        save_progress(progress_file, {"completed": sorted(keys)})
        loaded = load_progress(progress_file)
        assert set(loaded["completed"]) == set(keys)


class TestGroupRacekeysByDate:
    """racekey の日付グルーピングテスト"""

    def test_basic_grouping(self):
        racekeys = [
            "202602030501",
            "202602030502",
            "202602030901",
            "202602040501",
        ]
        grouped = group_racekeys_by_date(racekeys)

        assert len(grouped) == 2
        assert "20260203" in grouped
        assert "20260204" in grouped
        assert len(grouped["20260203"]) == 3
        assert len(grouped["20260204"]) == 1

    def test_empty_list(self):
        assert group_racekeys_by_date([]) == {}

    def test_single_date(self):
        racekeys = ["202602030501", "202602030502"]
        grouped = group_racekeys_by_date(racekeys)
        assert len(grouped) == 1
        assert len(grouped["20260203"]) == 2


class TestOpsDefaultDataspecs:
    """ops デフォルト値のテスト"""

    def test_contains_all_required(self):
        # 必須 dataspecs が含まれていること
        assert "0B41" in OPS_DEFAULT_DATASPECS
        assert "0B11" in OPS_DEFAULT_DATASPECS
        assert "0B16" in OPS_DEFAULT_DATASPECS
        assert "0B13" in OPS_DEFAULT_DATASPECS
        assert "0B17" in OPS_DEFAULT_DATASPECS

    def test_count(self):
        assert len(OPS_DEFAULT_DATASPECS) == 5


class TestToWindowsPath:
    """WSL → Windows パス変換のテスト（フォールバック）"""

    def test_mnt_path_fallback(self, monkeypatch):
        """wslpath が使えない場合の /mnt/c/ 変換"""
        from scripts import rt_common

        # wslpath を使えなくする
        original = rt_common.subprocess.run

        def mock_run(cmd, **kwargs):
            if cmd[0] == "wslpath":
                raise FileNotFoundError
            return original(cmd, **kwargs)

        monkeypatch.setattr(rt_common.subprocess, "run", mock_run)

        result = rt_common._to_windows_path("/mnt/c/Users/sato/data")
        assert result == "C:\\Users\\sato\\data"

    def test_wsl_native_path_fallback(self, monkeypatch):
        """wslpath が使えない場合の WSL ネイティブパス → UNC 変換"""
        from scripts import rt_common

        original = rt_common.subprocess.run

        def mock_run(cmd, **kwargs):
            if cmd[0] == "wslpath":
                raise FileNotFoundError
            return original(cmd, **kwargs)

        monkeypatch.setattr(rt_common.subprocess, "run", mock_run)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

        result = rt_common._to_windows_path("/home/sato/projects/keibas/data")
        assert result == "\\\\wsl.localhost\\Ubuntu\\home\\sato\\projects\\keibas\\data"

    def test_wslpath_integration(self):
        """wslpath コマンドが利用可能な場合の実際の変換"""
        from scripts.rt_common import _to_windows_path

        result = _to_windows_path("/mnt/c/tmp")
        # wslpath があれば C:\tmp、なければフォールバック
        assert "tmp" in result
        assert result.startswith("C:") or result.startswith("\\\\")
