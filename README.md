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
│   ├── 02_char_tfidf_logreg.ipynb
│   └── 03_generate_embeddings.ipynb
├── tests/
│   ├── test_validation.py
│   ├── test_text_features.py
│   └── test_embedding_features.py
├── requirements.txt
├── embedding_features.py      # 外部Embedding API・resume・保存
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

## 外部Embedding API

`embedding_features.py`はOpenAIとGoogle Geminiに対応し、生成物を`data/embeddings/`へ保存します。APIキーは環境変数から読み込み、コード・ログ・保存ファイルには含めません。

```powershell
$env:OPENAI_API_KEY="..."
# または
$env:GEMINI_API_KEY="..."
```

最初に`notebooks/03_generate_embeddings.ipynb`のdry-runで、行数、文字数、推定token、推定費用、truncate候補を確認します。少量の疎通確認は`RUN_SMOKE_API`、全件生成は`RUN_FULL_API`を明示的に`True`へ変更した場合だけ実行します。

```text
data/embeddings/
└── openai_text-embedding-3-large_dim1536_<config-hash>/
    ├── config.json
    ├── train_embeddings.npy
    ├── train_metadata.parquet
    ├── train_progress.json
    ├── test_embeddings.npy
    ├── test_metadata.parquet
    ├── test_progress.json
    └── shards/
```

batchごとにshardを保存するため、中断後は取得済みbatchを再利用して続きから再開します。provider、model、dimension、使用列、template、正規化、text hashが一致しないcacheは再利用しません。

既定モデルはOpenAIが`text-embedding-3-large`、Geminiが`gemini-embedding-2`です。次元数は既定で1536、`None`ならモデル既定値を使用します。Gemini Embedding 2の分類用途は、現在のAPI仕様に合わせてテキスト先頭へtask prefixを付けます。利用可能モデルを確認したい場合は`list_embedding_models()`を使い、利用不能時に別モデルへ自動fallbackはしません。

```python
from embedding_features import generate_embeddings

result = generate_embeddings(
    df=train,
    split="train",
    provider="openai",
    model="text-embedding-3-large",
    embedding_dim=1536,
    output_root="data/embeddings",
    dry_run=True,  # APIは呼ばない
)
```
