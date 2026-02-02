#!/bin/bash
set -e

# 1. PostgreSQLのインストール
echo "PostgreSQLをインストールしています..."
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# 2. サービスの起動
echo "PostgreSQLサービスを起動しています..."
sudo service postgresql start
sudo service postgresql status

# 3. ユーザーとデータベースの作成
echo "ユーザーとデータベースを作成しています..."
# デフォルトパスワード 'keiba_pass' を設定（必要に応じて変更してください）
DB_PASS="keiba_pass"

# 既に存在しない場合のみユーザーを作成
sudo -u postgres psql -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'jv_ingest') THEN CREATE ROLE jv_ingest WITH LOGIN PASSWORD '${DB_PASS}'; END IF; END \$\$;"

# 既に存在しない場合のみDBを作成
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = 'keiba'" | grep -q 1; then
    echo "keiba データベースを作成しています..."
    sudo -u postgres createdb -O jv_ingest keiba
else
    echo "keiba データベースは既に存在します。"
fi

# 権限付与
sudo -u postgres psql -c "GRANT CREATE, CONNECT ON DATABASE keiba TO jv_ingest;"

# 4. 接続設定
echo "接続設定を行っています..."
# Mirrored Networking Mode (WSL2の最近のデフォルト) を想定し、localhost (127.0.0.1) を許可します。
HBA_FILE=$(sudo -u postgres psql -c "SHOW hba_file;" -tA)
echo "pg_hba.conf の場所: $HBA_FILE"

# jv_ingest 用の設定が存在しなければ追加
if ! sudo grep -q "host.*keiba.*jv_ingest.*127.0.0.1" "$HBA_FILE"; then
    echo "pg_hba.conf にルールを追加しています..."
    echo "host    keiba           jv_ingest           127.0.0.1/32            scram-sha-256" | sudo tee -a "$HBA_FILE"
else
    echo "pg_hba.conf に既に jv_ingest 用のルールが存在します。"
fi

# 設定の再読み込み
sudo service postgresql reload

echo "セットアップが完了しました！"
echo "データベース: keiba"
echo "ユーザー: jv_ingest"
echo "パスワード: ${DB_PASS}"
