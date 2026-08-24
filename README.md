# Competition workspace

データ分析コンペで、データの読込、時系列CVの確認、モデル実験をすぐ始めるための最小構成です。

```text
competition/
├── data/
│   └── csv/                    # 生成特徴量・OOF・実験結果（Git管理対象外）
├── input/
│   ├── train.csv              # Git管理対象外
│   └── test.csv               # Git管理対象外
├── notebooks/
│   ├── 01_time_series_cv.ipynb
│   └── 02_char_tfidf_logreg.ipynb
├── tests/
│   ├── test_validation.py
│   └── test_text_features.py
├── requirements.txt
├── text_features.py           # char TF-IDF・Logistic Regression
├── validation.py              # CV・seen/unseen判定
└── README.md
```

## 使い方

1. `input/`に`train.csv`や`test.csv`を配置します。CSVは`.gitignore`によりGitへ追加されません。
2. `python -m pip install -r requirements.txt`で必要パッケージを用意します。
3. リポジトリのルート、または`notebooks/`からJupyterを起動します。
4. `notebooks/01_time_series_cv.ipynb`を開き、設定セルの列名を実データに合わせます。
5. 上から実行し、`diagnostics`で各foldの年度、件数、target平均、seen/unseen比率を確認します。
6. `02_char_tfidf_logreg.ipynb`で列別・全列結合のchar TF-IDFを評価します。

```powershell
jupyter lab
```

## 時系列CV

`validation.py`の`make_time_series_cv`は、`project_start_year`として実在する正常年度のうち、最新の3年度をvalidationにします。各foldのtrainingは常にvalidation年度より前の全行です。

```python
from validation import make_seen_project_mask, make_time_series_cv

folds, diagnostics = make_time_series_cv(
    df=train,
    year_col="project_start_year",
    project_col="project_name",
    target_col="target",
    n_valid_years=3,
)

for train_idx, valid_idx in folds:
    # 返されるのは元DataFrameのラベルindexなので、必ずlocを使う。
    train_fold = train.loc[train_idx]
    valid_fold = train.loc[valid_idx]

    seen_project = make_seen_project_mask(
        df=train,
        train_idx=train_idx,
        valid_idx=valid_idx,
        project_col="project_name",
    )
```

主な仕様は次のとおりです。

- `project_start_year == -1`と解釈不能な年度は、元データを変更せずfoldからだけ除外
- `project_name`によるGroupKFoldやpurgeは行わない
- seen/unseenは、そのfoldのtrainingに存在する`project_name`だけから判定
- indexが非連番でも利用可能。ただし`.loc`を安全に使うためindexの重複は禁止
- `diagnostics["is_latest_fold"]`で最新年度のfoldを特定可能

## OOF prediction

validationにならない古い年度と異常年度の行を`NaN`のまま残します。NumPy配列へ格納するときは、ラベルindexを位置indexへ明示的に変換します。

```python
import numpy as np

oof = np.full(len(train), np.nan, dtype=float)

for train_idx, valid_idx in folds:
    # valid_prediction = model.predict_proba(train.loc[valid_idx, features])[:, 1]
    valid_positions = train.index.get_indexer(valid_idx)
    assert (valid_positions >= 0).all()
    # oof[valid_positions] = valid_prediction
```

## テスト

```powershell
python -m unittest discover -s tests -v
```

年度が欠けている場合、`-1`の除外、非連番index、seen/unseen判定、trainingが空のfoldのskipをテストしています。

## Char TF-IDF + Logistic Regression

`text_features.py`は、各foldのtraining部分だけで列別の`TfidfVectorizer`をfitします。validationの文章はIDFや語彙作成に使いません。

```python
from text_features import (
    compare_cv_results,
    cross_validate_text_columns,
)

combined_result = cross_validate_text_columns(
    df=train,
    folds=folds,
    text_cols=["project_name", "project_objective", "project_summary"],
    target_col="target",
    project_col="project_name",
    year_col="project_start_year",
    model_name="all_columns_char_tfidf",
    metric="roc_auc",
    feature_output_dir="data/csv",
)

display(compare_cv_results([combined_result]))
```

### 特徴量CSV

TF-IDFをdense化せず、非ゼロ要素だけを次のlong形式で保存します。

```text
row_index,feature_index,value
```

特徴名は同じフォルダの`feature_names.csv.gz`で`feature_index`と対応します。既定の出力例は次のとおりです。

```text
data/csv/
└── all_columns_char_tfidf/
    ├── fold_0_year_2020/
    │   ├── train_features.csv.gz
    │   ├── valid_features.csv.gz
    │   ├── feature_names.csv.gz
    │   └── validation_predictions.csv.gz
    ├── oof_predictions.csv.gz
    └── fold_scores.csv
```

CSVは非常に大きくなる可能性があります。`feature_output_dir=None`にすると保存を無効化できます。
