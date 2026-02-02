import psycopg
import sys

def check_connection():
    try:
        conn = psycopg.connect(
            host="127.0.0.1",
            dbname="keiba",
            user="jv_ingest",
            password="keiba_pass"
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 AS connected;")
        result = cur.fetchone()
        print(f"接続成功！ 結果: {result}")
        conn.close()
        return 0
    except Exception as e:
        print(f"接続失敗: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_connection())
