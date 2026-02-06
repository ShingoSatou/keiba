"""
JV-Link COM ラッパーモジュール

JRA-VAN Data Lab. の JV-Link を Python から操作するためのラッパークラス。
Windows環境でのみ動作します。

使用例:
    from app.infrastructure.jvlink import JVLinkClient

    with JVLinkClient() as jv:
        jv.open("RACE", "20260101", option=1)
        for record in jv.read_all():
            print(record.rec_id, record.payload[:50])
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from win32com.client import CDispatch

# Windows以外では動作しない
if sys.platform != "win32":
    raise ImportError("JVLink is only available on Windows")

import win32com.client


@dataclass
class JVRecord:
    """JVReadで取得した1レコードを表すデータクラス"""

    rec_id: str  # レコード種別 (2文字: RA, SE, HR, etc.)
    filename: str  # JVReadから返されたファイル名
    payload: str  # 固定長文字列データ (Shift_JIS→UTF-8変換済み)


class JVLinkError(Exception):
    """JV-Link操作時のエラー"""

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(f"JVLink Error [{code}]: {message}")


class JVLinkClient:
    """
    JV-Link COM オブジェクトのラッパークラス

    コンテキストマネージャとして使用可能:
        with JVLinkClient() as jv:
            jv.open(...)
            for record in jv.read_all():
                ...
    """

    # JVOpen の option フラグ (JRA-VAN仕様準拠)
    # 蓄積系データ (RACE等)
    OPTION_NORMAL = 1  # 通常データ取得 (過去1年分、日々の更新用)
    OPTION_SETUP = 3  # セットアップ (過去データ一括取得、ダイアログあり)
    OPTION_SETUP_WITHOUT_DIALOG = 4  # セットアップ (ダイアログなし、推奨)
    # 非蓄積系データ (速報等)
    OPTION_REALTIME = 2  # 非蓄積系データ取得

    # JVRead の戻り値
    READ_SUCCESS = 0  # 全データ読み込み完了
    READ_ERROR_NOT_OPEN = -1
    READ_ERROR_FILE = -2
    READ_ERROR_DOWNLOAD = -3

    def __init__(self, sid: str = "UNKNOWN"):
        """
        JVLinkクライアントを初期化

        Args:
            sid: ソフトウェアID (JRA-VANから発行されたID、未登録の場合は "UNKNOWN")
        """
        self._sid = sid
        self._jv: CDispatch | None = None
        self._is_open = False

    def __enter__(self) -> JVLinkClient:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def init(self) -> int:
        """
        JV-Link を初期化

        Returns:
            0: 成功
            負値: エラーコード
        """
        self._jv = win32com.client.Dispatch("JVDTLab.JVLink")
        result = self._jv.JVInit(self._sid)
        if result != 0:
            raise JVLinkError(result, "JVInit failed")
        return result

    def open(
        self,
        dataspec: str,
        from_date: str | datetime,
        option: int = OPTION_NORMAL,
        key: str = "",
        outside_ref: str = "",
    ) -> tuple[int, int]:
        """
        データの読み込みを開始

        Args:
            dataspec: データ種別 (例: "RACE", "DIFF", "BLOD" など)
            from_date: 取得開始日 (YYYYMMDD形式の文字列 or datetime)
            option: オプションフラグ (OPTION_NORMAL, OPTION_SETUP, etc.)
            key: 検索キー (通常は空文字)
            outside_ref: 外部参照モード (通常は空文字)

        Returns:
            (read_count, download_count): 読み込み件数とダウンロード件数
        """
        if self._jv is None:
            raise JVLinkError(-100, "JVLink not initialized. Call init() first.")

        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y%m%d")

        # JVOpenの引数: dataspec, fromtime, option, key, outsideref, buffersize
        # buffersize は出力変数（読み込み件数）
        read_count = 0
        download_count = 0

        result = self._jv.JVOpen(
            dataspec, from_date, option, key, outside_ref, read_count, download_count
        )

        if result < 0:
            raise JVLinkError(result, f"JVOpen failed for dataspec={dataspec}")

        self._is_open = True
        # JVOpenの戻り値からread_count, download_countを取得
        # ※ COMの仕様上、outパラメータは別途取得が必要な場合あり
        return (result, 0)

    def read(self) -> tuple[int, JVRecord | None]:
        """
        1レコード読み込み

        Returns:
            (status, record):
                status > 0: データあり (status = バイト数)
                status == 0: 読み込み完了
                status < 0: エラー
        """
        if self._jv is None or not self._is_open:
            raise JVLinkError(-100, "JVLink not open. Call open() first.")

        # JVRead の引数: buff(out), size(out), fname(out)
        # Pythonからは戻り値として受け取る
        buff = ""
        size = 0
        fname = ""

        result = self._jv.JVRead(buff, size, fname)

        if result > 0:
            # result = 読み込んだバイト数
            # buff には Shift_JIS のデータが入る（COMの仕様による）
            # 実際のbuffはタプルで返される可能性あり
            actual_buff = buff if isinstance(buff, str) else str(buff)
            rec_id = actual_buff[:2] if len(actual_buff) >= 2 else ""
            record = JVRecord(rec_id=rec_id, filename=fname, payload=actual_buff)
            return (result, record)
        elif result == 0:
            return (0, None)  # 読み込み完了
        elif result == -1:
            # ファイル切り替え（JVGoto必要）
            return (-1, None)
        else:
            raise JVLinkError(result, "JVRead failed")

    def read_all(self) -> Iterator[JVRecord]:
        """
        全レコードをイテレートして返すジェネレータ

        Yields:
            JVRecord: 読み込んだレコード
        """
        while True:
            status, record = self.read()
            if status > 0 and record is not None:
                yield record
            elif status == 0:
                break  # 読み込み完了
            elif status == -1:
                # ファイル切り替え時はスキップして続行
                continue
            else:
                break  # エラー

    def close(self) -> None:
        """JV-Link をクローズ"""
        if self._jv is not None and self._is_open:
            self._jv.JVClose()
            self._is_open = False

    def status(self) -> dict:
        """
        JV-Link のステータス情報を取得

        Returns:
            {"version": ..., "last_update": ..., ...}
        """
        if self._jv is None:
            raise JVLinkError(-100, "JVLink not initialized")

        # JVStatus で最終更新日時などを取得
        # 実装は仕様書に従って調整が必要
        return {"initialized": True}


# DataSpec 定数（よく使うもの）
class DataSpec:
    """JVOpenで指定するデータ種別"""

    # 蓄積系（過去データ）
    RACE = "RACE"  # レース詳細 (RA/SE/HR/H1/H6/O1-O6)
    DIFF = "DIFF"  # 差分データ
    BLOD = "BLOD"  # 血統データ
    MING = "MING"  # マイニング（セットアップ用）
    SNAP = "SNAP"  # スナップショット

    # 速報系
    TCOV = "TCOV"  # 速報成績
    RCOV = "RCOV"  # レース速報


if __name__ == "__main__":
    # 簡易テスト（JV-Linkがインストールされている環境で実行）
    print("JVLink module loaded successfully")
    print(f"Platform: {sys.platform}")
