# データ設計ドキュメント

## 概要
JRA-VAN Data Lab. (JV-Link) から取得するデータの設計方針。

---

## 1. 取得データ対象

### 1.1 実装済みレコード
| Record | 説明 | パーサー | DB |
|--------|------|----------|-------|
| **RA** | レース詳細 | ✅ `RaceRecord` | ✅ `core.race` |
| **SE** | 出走馬/結果 | ✅ `RunnerRecord` | ✅ `core.runner` |
| **HR** | 払戻金 | ✅ `PayoutRecord` | ✅ `core.payout` |
| **O1** | オッズ (単複枠) | ✅ `OddsRecord` | ✅ `core.odds_final` |
| **JG** | 競走馬除外 | ✅ `HorseExclusionRecord` | ✅ `core.horse` |
| **UM** | 馬マスタ | ✅ `HorseMasterRecord` | ✅ `core.horse` |
| **KS** | 騎手マスタ | ✅ `JockeyRecord` | ✅ `core.jockey` |
| **CH** | 調教師マスタ | ✅ `TrainerRecord` | ✅ `core.trainer` |

---

## 2. パーサー実装

### 2.1 バイトオフセット (重要)
JV-Dataは**バイト位置**で定義。仕様書は1-indexed、コードは0-indexed。

```python
# 例: RAレコード (JV-Data 4.9.0.1準拠)
RA_YEAR_START = 11       # 仕様書: 12
RA_DISTANCE_START = 697  # 仕様書: 698
RA_TRACK_TYPE_START = 705  # 仕様書: 706
RA_WEATHER_START = 887   # 仕様書: 888
```

### 2.2 Surface判定 (コード表2009)
```python
# 正しい範囲
if 23 <= track_type_code <= 29:
    surface = 2  # ダート
elif 51 <= track_type_code <= 59:
    surface = 3  # 障害
else:
    surface = 1  # 芝 (デフォルト)
```

### 2.3 パーサー呼び出し例
```python
from app.infrastructure.parsers import RaceRecord, RunnerRecord

# RA: 単一オブジェクト
race = RaceRecord.parse(payload)

# SE: 単一オブジェクト
runner = RunnerRecord.parse(payload)

# HR: リスト
payouts = PayoutRecord.parse(payload)  # list[PayoutRecord]
```

---

## 3. データベース設計

### 3.1 スキーマ構成
| スキーマ | 役割 |
|----------|------|
| `raw` | 生データ保持（再パース用） |
| `core` | 正規化テーブル |
| `mart` | 特徴量テーブル（将来） |

### 3.2 主キー設計

#### race_id
```
race_id = YYYYMMDD * 10000 + track_code * 100 + race_no
例: 2026/02/03 東京(05) 5R → 202602030505
```

#### horse_id
```
horse_id = TEXT型 (血統登録番号)
```

### 3.3 テーブル一覧
| テーブル | PK | 状態 |
|----------|-------|------|
| `core.race` | race_id | ✅ |
| `core.runner` | (race_id, horse_id) | ✅ |
| `core.horse` | horse_id | ✅ |
| `core.jockey` | jockey_id | ✅ |
| `core.trainer` | trainer_id | ✅ |
| `core.payout` | (race_id, bet_type, selection) | ✅ |
| `core.odds_final` | (race_id, horse_id) | ✅ |

### 3.4 FK問題と解決策 (実装済み)

**問題**: 地方競馬/海外データはマスタが欠落

**解決策**:
1. `jockey_id`, `trainer_id` を NULL許容
2. `core.horse`, `core.race` へ自動Stub登録
3. 生値カラム (`trainer_code_raw`, `jockey_code_raw` 等) で元データ保持

---

## 4. 検証ツール

```powershell
# パーサーテスト
uv run pytest tests/test_parsers.py -v

# DB統計確認
uv run python scripts/analyze_db_stats.py

# スキーマ確認
uv run python scripts/check_schema.py
```

---

## 5. 参考資料
- `docs/JV-link/JV-Data仕様書_4.9.0.1.xlsx` (コード表含む)
- `docs/JV-link/JV-Linkインターフェース仕様書_4.9.0.1(Win).pdf`
