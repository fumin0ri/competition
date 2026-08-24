# Competition workspace

データ分析コンペで、データの読込、時系列CVの確認、モデル実験をすぐ始めるための最小構成です。

```text
competition/
├── input/
│   ├── train.csv              # Git管理対象外
│   └── test.csv               # Git管理対象外
├── notebooks/
│   └── 01_time_series_cv.ipynb
├── tests/
│   └── test_validation.py
├── validation.py              # CV・seen/unseen判定
└── README.md
```

## 使い方

1. `input/`に`train.csv`や`test.csv`を配置します。CSVは`.gitignore`によりGitへ追加されません。
2. リポジトリのルート、または`notebooks/`からJupyterを起動します。
3. `notebooks/01_time_series_cv.ipynb`を開き、設定セルの列名を実データに合わせます。
4. 上から実行し、`diagnostics`で各foldの年度、件数、target平均、seen/unseen比率を確認します。

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
    oof[valid_positions] = valid_prediction
```

## テスト

```powershell
python -m unittest discover -s tests -v
```

年度が欠けている場合、`-1`の除外、非連番index、seen/unseen判定、trainingが空のfoldのskipをテストしています。
