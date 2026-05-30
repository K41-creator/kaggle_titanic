# Titanic Survival Prediction

## 概要

Kaggle の Titanic コンペティション「Titanic - Machine Learning from Disaster」に参加し、乗客情報から生存者を予測する機械学習モデルを作成しました。

このプロジェクトでは、Python・pandas・scikit-learn を用いてデータ分析、前処理、モデル学習、予測を行いました。

---

## 使用技術

* Python 3.13
* pandas
* scikit-learn
* NumPy

---

## コンペティション概要

タイタニック号の乗客データを用いて、各乗客が生存したかどうかを予測する二値分類問題です。

予測対象：

* 1 = 生存
* 0 = 死亡

評価指標：

* Accuracy（正解率）

---

## 使用した特徴量

* Pclass（客室クラス）
* Sex（性別）
* SibSp（兄弟・配偶者数）
* Parch（親・子供数）

カテゴリ変数である Sex は One-Hot Encoding を用いて数値化しました。

---

## モデル

RandomForestClassifier

主な設定：

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

## 前処理

### 特徴量選択

学習に使用する列を選択

```python
features = [
    "Pclass",
    "Sex",
    "SibSp",
    "Parch"
]
```

### カテゴリ変数の変換

```python
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
```

---

## 結果

初回提出スコア：

**0.77272**

---

## 学んだこと

* pandas を用いたデータ操作
* DataFrame と Series の違い
* groupby による集計
* One-Hot Encoding
* Random Forest の基本
* Kaggle への提出方法
* 機械学習プロジェクトの基本的な流れ

---

## 今後の改善案

* Age の欠損値補完
* Fare の利用
* FamilySize 特徴量の追加
* Feature Importance の分析
* ハイパーパラメータチューニング
* 交差検証の導入

---

## 実行方法

必要ライブラリのインストール

```bash
pip install pandas numpy scikit-learn
```

実行

```bash
python main.py
```

予測結果は submission.csv として出力されます。

---

## 参考

* Kaggle Titanic Competition
* pandas Documentation
* scikit-learn Documentation
# kaggle_titanic


---

## 改善ログ（更新）

### 追加した特徴量
- Embarked（乗船港）を追加
- FamilySize（SibSp + Parch + 1）を追加

### 欠損値処理の改善
- Age：中央値で補完
- Fare：中央値で補完（testデータ）
- Embarked：最頻値で補完

### モデル改善
- RandomForestClassifierのパラメータを調整
  - n_estimators = 200
  - max_depth = 5
  - random_state = 42

### 評価方法
- train_test_splitによるホールドアウト検証を実施
- accuracy_scoreで精度評価

### 最新スコア
- Kaggle Public Score: **0.78708**

### 学んだこと
- 特徴量エンジニアリングの重要性
- 欠損値処理の影響
- 過学習と汎化性能の関係
- Random Forestの挙動理解
- train/test分割と評価の意味

---

## 今後の改善予定
- NameからTitle（Mr, Mrs, Missなど）の抽出
- モデルのハイパーパラメータ調整
- Cross Validationの導入
- 特徴量の重要度分析による最適化