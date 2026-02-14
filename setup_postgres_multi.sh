#!/usr/bin/env bash
set -euo pipefail

# ===== 設定 =====
DB_USER="${DB_USER:-jv_ingest}"
DBS_DEFAULT=("keiba_cp" "keiba_ag" "keiba_main")
DBS=("${DBS_DEFAULT[@]}")

# 引数でDB名を上書きできる（例: ./setup_postgres_multi.sh keiba_cp keiba_ag）
if [[ $# -ge 1 ]]; then
  DBS=("$@")
fi

echo "[INFO] DB_USER = ${DB_USER}"
echo "[INFO] DBS     = ${DBS[*]}"

# パスワードは入力させる（履歴に残さない）
if [[ -z "${DB_PASS:-}" ]]; then
  read -r -s -p "Password for role '${DB_USER}': " DB_PASS
  echo
fi

# ===== 1) PostgreSQL install (既に入っていればスキップ) =====
if ! dpkg -s postgresql >/dev/null 2>&1; then
  echo "[INFO] Installing PostgreSQL..."
  sudo apt update
  sudo apt install -y postgresql postgresql-contrib
else
  echo "[INFO] PostgreSQL already installed."
fi

# ===== 2) service start =====
echo "[INFO] Starting PostgreSQL service..."
sudo service postgresql start >/dev/null
sudo service postgresql status || true

# ===== 3) role create (if not exists) =====
echo "[INFO] Creating role if not exists..."
sudo -u postgres psql -v ON_ERROR_STOP=1 -c \
"DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';
  END IF;
END \$\$;"

# ===== 4) db create (if not exists) + grant =====
for DB_NAME in "${DBS[@]}"; do
  echo "[INFO] Ensure database: ${DB_NAME}"
  if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1; then
    sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
    echo "[INFO] created: ${DB_NAME}"
  else
    echo "[INFO] exists : ${DB_NAME}"
  fi

  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "GRANT CONNECT, CREATE ON DATABASE ${DB_NAME} TO ${DB_USER};"
done

# ===== 5) pg_hba.conf allow localhost for this user =====
HBA_FILE=$(sudo -u postgres psql -tA -c "SHOW hba_file;")
echo "[INFO] pg_hba.conf = ${HBA_FILE}"

# IPv4 localhost
RULE_V4="host    all     ${DB_USER}     127.0.0.1/32     scram-sha-256"
if ! sudo grep -qF "$RULE_V4" "$HBA_FILE"; then
  echo "[INFO] append pg_hba (v4)"
  echo "$RULE_V4" | sudo tee -a "$HBA_FILE" >/dev/null
fi

# IPv6 localhost
RULE_V6="host    all     ${DB_USER}     ::1/128          scram-sha-256"
if ! sudo grep -qF "$RULE_V6" "$HBA_FILE"; then
  echo "[INFO] append pg_hba (v6)"
  echo "$RULE_V6" | sudo tee -a "$HBA_FILE" >/dev/null
fi

sudo service postgresql reload
echo "[OK] Setup complete."
echo "  role : ${DB_USER}"
echo "  dbs  : ${DBS[*]}"
