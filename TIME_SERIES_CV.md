# 時系列Cross Validation

## 1. CV設計の考え方

`project_start_year`として実在する正常年度のうち、最新の3年度をvalidationにします。各foldは、validation年度より前の全データをtrainingに含めるexpanding windowです。

- `project_start_year == -1`と数値として解釈できない年度は、元データを変更せずfold indexからだけ除外します。
- `project_name`によるGroupKFoldやpurgeは行いません。
- seen/unseenは、各foldのtrainingに同名の`project_name`が存在するかを完全一致で判定します。未来側の行は参照しません。
- 返すindexは元DataFrameのラベルindexです。必ず`.loc`で使用します。そのため、元DataFrameのindexには一意性を要求します。
- `diagnostics["is_latest_fold"]`で、最新年度のfoldを簡単に特定できます。

## 2. 完成したPythonコード

実装本体は[`time_series_cv.py`](./time_series_cv.py)です。公開APIは次の2関数です。

```python
from time_series_cv import make_seen_project_mask, make_time_series_cv
```

- `make_time_series_cv`: expanding-window foldとfold単位の診断表を作成
- `make_seen_project_mask`: 任意foldのvalidation各行をseen/unseenに分解

## 3. 使用例

```python
from time_series_cv import make_seen_project_mask, make_time_series_cv

folds, diagnostics = make_time_series_cv(
    df=train,
    year_col="project_start_year",
    project_col="project_name",
    target_col="target",
    n_valid_years=3,
)

feature_cols = [
    col for col in train.columns
    if col not in {"target", "project_id"}
]

for fold, (train_idx, valid_idx) in enumerate(folds):
    # train_idx / valid_idxはラベルindexなので、ilocではなくlocを使う。
    X_train = train.loc[train_idx, feature_cols]
    y_train = train.loc[train_idx, "target"]
    X_valid = train.loc[valid_idx, feature_cols]
    y_valid = train.loc[valid_idx, "target"]

    seen_project = make_seen_project_mask(
        df=train,
        train_idx=train_idx,
        valid_idx=valid_idx,
        project_col="project_name",
    )

    print(
        f"fold={fold}",
        f"train={len(train_idx)}",
        f"valid={len(valid_idx)}",
        f"seen={seen_project.sum()}",
        f"unseen={(~seen_project).sum()}",
    )
```

`seen_project`は`valid_idx`と同じラベルindexを持つBoolean Seriesです。後で予測値を同じindexのSeriesにすれば、全体・seen・unseenの評価を安全に分けられます。

```python
from sklearn.metrics import roc_auc_score

# valid_prediction = fitted_model.predict_proba(X_valid)[:, 1]
prediction = pd.Series(valid_prediction, index=valid_idx, name="prediction")
actual = train.loc[valid_idx, "target"]

scores = {"all": roc_auc_score(actual, prediction)}
if actual.loc[seen_project].nunique() == 2:
    scores["seen"] = roc_auc_score(
        actual.loc[seen_project], prediction.loc[seen_project]
    )
if actual.loc[~seen_project].nunique() == 2:
    scores["unseen"] = roc_auc_score(
        actual.loc[~seen_project], prediction.loc[~seen_project]
    )
```

## 4. DiagnosticsをDataFrameで確認する例

```python
display(diagnostics)

latest_fold = diagnostics.loc[diagnostics["is_latest_fold"]]
display(latest_fold)
```

`diagnostics`には以下が入ります。

- `validation_year`
- `train_size`
- `validation_size`
- `validation_target_mean`
- `train_target_mean`
- `seen_project_rate`
- `seen_count`
- `unseen_count`
- `is_latest_fold`

## 5. OOF predictionを保存するサンプル

元DataFrameのindexが0始まりの連番とは限らないため、NumPy配列へ代入するときはラベルindexを位置へ明示的に変換します。validationにならない古い年度と`project_start_year == -1`の行は`NaN`のまま残ります。

```python
import numpy as np

oof = np.full(len(train), np.nan, dtype=float)

for fold, (train_idx, valid_idx) in enumerate(folds):
    X_train = train.loc[train_idx, feature_cols]
    y_train = train.loc[train_idx, "target"]
    X_valid = train.loc[valid_idx, feature_cols]

    # fitted_model = ...
    # fitted_model.fit(X_train, y_train)
    # valid_prediction = fitted_model.predict_proba(X_valid)[:, 1]

    valid_positions = train.index.get_indexer(valid_idx)
    assert (valid_positions >= 0).all()
    oof[valid_positions] = valid_prediction

evaluated_mask = ~np.isnan(oof)
print("OOF evaluated rows:", evaluated_mask.sum())
print("OOF unevaluated rows:", (~evaluated_mask).sum())
```

ラベルindexのまま管理したい場合は、次のSeries形式がより単純です。

```python
oof_series = pd.Series(np.nan, index=train.index, name="oof_prediction")
oof_series.loc[valid_idx] = valid_prediction
```
