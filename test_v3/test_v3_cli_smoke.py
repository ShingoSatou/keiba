import subprocess
from pathlib import Path


def run_cli_help(script_name: str) -> None:
    script_path = Path(f"scripts_v3/{script_name}")
    assert script_path.exists(), f"{script_path} does not exist"

    result = subprocess.run(
        ["uv", "run", "python", str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{script_name} --help failed: {result.stderr}"
    assert "usage:" in result.stdout.lower()


def test_train_binary_model_help() -> None:
    run_cli_help("train_binary_model_v3.py")


def test_train_pl_help() -> None:
    run_cli_help("train_pl_v3.py")


def test_backtest_wide_help() -> None:
    run_cli_help("backtest_wide_v3.py")


def test_train_odds_calibrator_help() -> None:
    run_cli_help("train_odds_calibrator_v3.py")


def test_predict_race_help() -> None:
    run_cli_help("predict_race_v3.py")
