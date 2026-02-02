"""
Windows側からWSL上のPostgreSQLへの接続テスト

使用方法:
  1. Windows側でpsycopgをインストール:
     pip install psycopg[binary]
  
  2. このスクリプトを実行:
     python check_db_connection_windows.py

  ※ Mirrored Networking Mode (WSL2のデフォルト) を想定し、
     localhost (127.0.0.1) で接続します。
     接続できない場合は、PowerShellで `wsl hostname -I` を実行し、
     そのIPアドレスに変更してお試しください。
"""

import psycopg
import sys


def check_connection():
    # Mirrored modeの場合は localhost で接続可能
    # 接続できない場合は `wsl hostname -I` で取得したIPに変更
    host = "127.0.0.1"
    port = 5432
    dbname = "keiba"
    user = "jv_ingest"
    password = "keiba_pass"

    print(f"接続先: {host}:{port}/{dbname} (user={user})")

    try:
        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 AS connected;")
        result = cur.fetchone()
        print(f"接続成功！ 結果: {result}")
        conn.close()
        return 0
    except Exception as e:
        print(f"接続失敗: {e}")
        print("")
        print("--- トラブルシュート ---")
        print("1. WSL側でPostgreSQLが起動しているか確認:")
        print("   sudo service postgresql status")
        print("")
        print("2. Mirrored modeでない場合、WSLのIPを確認:")
        print("   wsl hostname -I")
        print("   → 取得したIPを host= に設定して再試行")
        return 1


if __name__ == "__main__":
    sys.exit(check_connection())
