"""
当日運用 RACE 取込スクリプト。

対象日の RACE を抽出して JSONL を生成し、DB へロードする。
T-5運用で core.race / core.runner を確実に揃えるための運用コマンド。
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rt_common import DATA_DIR, detect_python32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _to_windows_path(posix_path: str) -> str:
    try:
        result = subprocess.run(
            ["wslpath", "-w", posix_path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    if posix_path.startswith("/mnt/"):
        parts = posix_path.split("/")
        if len(parts) >= 4 and parts[1] == "mnt":
            drive = parts[2].upper()
            rest = "\\".join(parts[3:])
            return f"{drive}:\\{rest}"
    return posix_path


def _find_existing_race_file(output_dir: Path, race_date: str) -> Path | None:
    pattern = f"RACE_{race_date}-{race_date}_*.jsonl"
    files = sorted(output_dir.glob(pattern))
    return files[-1] if files else None


def _run_command(cmd: list[str], dry_run: bool) -> None:
    logger.info("run: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_ops_race(
    race_date: str,
    python32: str,
    output_dir: Path,
    option: int,
    force: bool,
    dry_run: bool,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_race_file(output_dir, race_date)
    if existing and not force:
        logger.info("reuse existing RACE file: %s", existing)
        race_file = existing
    else:
        win_script_path = _to_windows_path(str(PROJECT_ROOT / "scripts" / "extract_jvlink.py"))
        win_output_dir = _to_windows_path(str(output_dir))
        cmd = [
            python32,
            win_script_path,
            "--dataspec",
            "RACE",
            "--from-date",
            race_date,
            "--to-date",
            race_date,
            "--option",
            str(option),
            "--output-dir",
            win_output_dir,
        ]
        logger.info("extract RACE: date=%s", race_date)
        if dry_run:
            logger.info("[DRY-RUN] %s", " ".join(cmd))
            return existing

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            capture_output=True,
            env=env,
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr or stdout or f"exit={result.returncode}"
            raise RuntimeError(f"RACE extract failed: {detail}")

        race_file = _find_existing_race_file(output_dir, race_date)
        if race_file is None:
            raise FileNotFoundError(
                "RACE extract finished but output file not found: "
                f"dir={output_dir} date={race_date}"
            )

    _run_command(
        ["uv", "run", "python", "scripts/load_to_db.py", "--input", str(race_file)],
        dry_run=dry_run,
    )
    return race_file


def main() -> None:
    parser = argparse.ArgumentParser(description="当日運用 RACE 抽出 + DB投入")
    parser.add_argument(
        "--race-date",
        default=None,
        help="対象日 (YYYYMMDD, デフォルト: 当日)",
    )
    parser.add_argument(
        "--python32",
        default=None,
        help="Windows 32bit Python パス (未指定時は自動検出)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"JSONL 出力先 (デフォルト: {DATA_DIR})",
    )
    parser.add_argument(
        "--option",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="extract_jvlink.py の option (デフォルト: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存 RACE JSONL があっても再取得する",
    )
    parser.add_argument("--dry-run", action="store_true", help="DRY-RUN")
    args = parser.parse_args()

    race_date = args.race_date or datetime.now().strftime("%Y%m%d")
    python32 = args.python32 or detect_python32()

    logger.info("ops_race start: date=%s output=%s", race_date, args.output_dir)
    race_file = run_ops_race(
        race_date=race_date,
        python32=python32,
        output_dir=args.output_dir,
        option=args.option,
        force=args.force,
        dry_run=args.dry_run,
    )
    logger.info("ops_race done: race_file=%s", race_file)


if __name__ == "__main__":
    main()
