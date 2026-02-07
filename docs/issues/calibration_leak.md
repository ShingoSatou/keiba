# 校正リーク問題 (Calibration Leak)

## 発見日
2026-02-07

## 概要

`train.py` において、確率校正器 (IsotonicRegression) がテストデータで「過学習」しており、
「完璧な校正」という誤った結論が出ている。

---

## 問題のコード

```python
# train.py train_model() 内

# 1. Train/Test 分割
X_train, X_test = ...  # 時系列で80/20分割
y_train, y_test = ...

# 2. モデル学習
lgb_model.fit(
    X_train, y_train,
    X_val=X_test,     # ← 問題1: TestをValidationに使用
    y_val=y_test,
    ...
)

# 3. 校正器学習
y_pred_raw = lgb_model.predict_proba(X_test)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(y_pred_raw, y_test)  # ← 問題2: Testで校正器を学習

# 4. 校正器評価
y_pred_calibrated = calibrator.predict(y_pred_raw)  # ← 問題3: 同じTestで評価
logloss_cal = log_loss(y_test, y_pred_calibrated)   # 当然「完璧」になる
```

---

## 問題点

### 問題1: Test = Validation の兼用
- Early Stopping の判定に `X_test` を使用
- これにより、モデルが間接的にテストデータに最適化されている

### 問題2: 校正器をテストデータで学習
- `calibrator.fit(y_pred_raw, y_test)` でテストデータの正解を使用
- 校正器がテストデータに「カンニング」している状態

### 問題3: 同じデータで評価
- 校正器を学習したデータで、校正器の性能を評価
- 当然「完璧」になる（過学習の典型例）

---

## 影響範囲

### 信頼できない指標
- 校正チェックログ（完璧に見える）
- 校正後 Logloss（改善したように見える）

### 信頼できる指標
- AUC（モデル自体の識別能力、校正とは独立）
- バックテスト結果（2025年以降の未見データで評価しているため）

---

## 修正方針

### 案A: 3分割方式（推奨）

```
[------- Train (60%) -------][-- Val (20%) --][-- Test (20%) --]
         ↓                        ↓                  ↓
    モデル学習              Early Stopping       最終評価
                            校正器学習
```

```python
# 時系列で3分割
split1 = int(len(df_sorted) * 0.6)
split2 = int(len(df_sorted) * 0.8)

train_idx = df_sorted.index[:split1]
val_idx = df_sorted.index[split1:split2]
test_idx = df_sorted.index[split2:]

# モデル学習 (Early Stopping は Val を使用)
lgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, ...)

# 校正器学習 (Val を使用)
y_pred_val = lgb_model.predict_proba(X_val)[:, 1]
calibrator.fit(y_pred_val, y_val)

# 最終評価 (Test を使用)
y_pred_test = lgb_model.predict_proba(X_test)[:, 1]
y_pred_calibrated = calibrator.predict(y_pred_test)
logloss_cal = log_loss(y_test, y_pred_calibrated)
```

### 案B: Cross-Validation 方式

学習データ内で K-Fold CV を行い、Out-of-Fold 予測値を使って校正器を学習。

---

## 修正後の期待される変化

1. **校正が「完璧」ではなくなる** → 予測0.10 → 実際0.12 程度のズレが出る
2. **Logloss 改善幅が小さくなる** → 0.2265 → 0.2260 程度
3. **バックテスト結果はほぼ変わらない** → 既に未見データで評価しているため

---

## 関連ファイル

- `scripts/train.py`: 修正対象
- `scripts/backtest.py`: 影響なし（別データで評価）

---

## 解決策 (2026-02-07 実装)

**3分割方式と校正器の廃止** を実施しました。

1. **データ分割**:
   - Train (70%) / ES-Val (10%) / Test (20%) に変更。
   - Test データは最終評価およびバックテストまで一切学習に使用しない。

2. **校正器の廃止**:
   - Isotonic/Sigmoid と校正なし (Raw) を比較した結果、Raw 予測の Logloss/ECE が最も良好であったため、**校正器を使用しない** 方針に決定。
   - これにより、Testデータを使った校正器学習（カンニング）の問題は根本的に解消された。

## ステータス

- [x] 3分割方式に修正
- [x] 校正器の比較検証 (None採用)
- [x] ドキュメント更新
- [x] **解決済み**


