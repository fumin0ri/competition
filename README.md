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
│   ├── 00_quick_eda.ipynb
│   ├── 01_time_series_cv.ipynb
│   ├── 02_char_tfidf_logreg.ipynb
│   ├── 03_generate_embeddings.ipynb
│   └── 04_embedding_models.ipynb
├── outputs/                    # モデルOOF・評価・test予測（Git管理対象外）
├── tests/
│   ├── test_validation.py
│   ├── test_text_features.py
│   ├── test_embedding_features.py
│   ├── test_modeling.py
│   └── test_ensemble.py
├── requirements.txt
├── embedding_features.py      # 外部Embedding API・resume・保存
├── ensemble.py                # AUC hill climbing・test blend・submission
├── modeling.py                # E1〜E6 / T1〜T3の統一比較
├── text_features.py           # char TF-IDF・Logistic Regression
├── validation.py              # CV・seen/unseen判定
└── README.md
```

## 使い方

1. `input/`に`train.csv`や`test.csv`を配置します。CSVは`.gitignore`によりGitへ追加されません。
2. `python -m pip install -r requirements.txt`で必要パッケージを用意します。
3. リポジトリのルート、または`notebooks/`からJupyterを起動します。
4. `notebooks/00_quick_eda.ipynb`を開き、設定セルの列名を実データに合わせて基本EDAを実行します。
5. `01_time_series_cv.ipynb`で各foldの年度、件数、target平均、seen/unseen比率を確認します。
6. `02_char_tfidf_logreg.ipynb`で列別・全列結合のchar TF-IDFを評価します。

```powershell
jupyter lab
```

## target列

このコンペのtarget列は`science_tech_decision`で、配布時は文字列ラベルです。各NotebookはCSV読込直後に共通関数で厳密に二値化します。

```python
from validation import encode_binary_target

TARGET_COL = "science_tech_decision"
train[TARGET_COL] = encode_binary_target(train[TARGET_COL])
# 該当 -> 1
# 非該当 -> 0
```

前後の空白は除去しますが部分一致は使いません。`非該当`に`該当`という文字列が含まれるためです。未知ラベルや欠損値は誤変換せずエラーにします。すでに0/1へ変換済みの場合も同じ関数を安全に再適用できます。

## Quick EDA

`notebooks/00_quick_eda.ipynb`はbaseline前に短時間でデータ全体を把握するためのNotebookです。次を表とグラフで確認します。

- train/testの行列数、型、重複、ID、列差
- target件数・比率
- train/testの欠損率
- 年度別件数、target率、クラス件数
- expanding-window CVの件数、target率、seen project率
- 数値列のtrain/test histogram、target別boxplot、Spearman相関
- カテゴリ上位構成比とカテゴリ別target率
- 日本語テキストの文字数、空文字率、target差、年度推移
- `project_name`の重複、target矛盾、train/test overlap
- 数値KS統計量・カテゴリtotal variationによる単変量のtrain/test差

描画処理だけ最大5万行へsamplingし、集計値は全行から計算します。列が存在しない場合は自動的に可視化対象から外れます。`SAVE_FIGURES=True`にすると、画像をGit管理対象外の`data/eda_figures/`へ保存します。モデルを使うadversarial validationはこのNotebookでは実行しません。

## 時系列CV

`validation.py`の`make_time_series_cv`は、`project_start_year`として実在する正常年度のうち、最新の3年度をvalidationにします。各foldのtrainingは常にvalidation年度より前の全行です。

```python
from validation import make_seen_project_mask, make_time_series_cv

folds, diagnostics = make_time_series_cv(
    df=train,
    year_col="project_start_year",
    project_col="project_name",
    target_col="science_tech_decision",
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
    target_col="science_tech_decision",
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

## Embedding + Tabular + TF-IDFの9実験

`modeling.py`は、既存の同一時系列foldを使って次を比較します。

| ID | 入力 | モデル |
|---|---|---|
| E1 | Embedding | Logistic Regression |
| E2 | Embedding + tabular | Logistic Regression |
| E3 | Embedding | PyTorch MLP |
| E4 | Embedding + tabular | PyTorch MLP |
| E5 | tabular | CatBoost |
| E6 | Embedding + tabular | XGBoost |
| T1 | char TF-IDF | Logistic Regression |
| T2 | char TF-IDF + tabular | Logistic Regression |
| T3 | char TF-IDF + Embedding + tabular | Logistic Regression |

T1〜T3は各foldで列別char TF-IDFを一度だけfitして共有します。validationは`transform`にしか使いません。T2/T3のtabular前処理もfold trainingだけでfitし、T3のEmbeddingはL2正規化後にCSR化します。全ブロックを`scipy.sparse.hstack`で結合するため、raw TF-IDFをdense化しません。

T4 x1を使う既定設定では、E3/E4がCUDA + mixed precision、E5が`task_type="GPU", devices="0"`、E6が`device="cuda", tree_method="hist"`です。LR、scikit-learn前処理、PCAはCPUで動きます。CPU環境では次のように変更できます。

```python
from modeling import default_modeling_config

CONFIG = default_modeling_config()
CONFIG["mlp"]["device"] = "cpu"
CONFIG["catboost"]["task_type"] = "CPU"
CONFIG["xgboost"]["device"] = "cpu"
```

コンペ指標に合わせ、early stoppingもROC-AUC基準です。CatBoostは`loss_function="Logloss", eval_metric="AUC"`、XGBoostは`objective="binary:logistic", eval_metric="auc"`を使います。MLPは`BCEWithLogitsLoss`で学習しつつ、`early_stop_metric="auc"`でbest epochを選びます。MLPのfold diagnosticsには選択epochの`best_validation_auc`と`best_validation_loss`を両方残します。比較目的でBCE基準へ戻す場合だけ、次を指定します。

```python
CONFIG["mlp"]["early_stop_metric"] = "loss"
```

E1/E2のLRは、denseなEmbeddingと中規模の特徴数に対する堅実なbaselineとして`lbfgs`を既定にしています。Embeddingがほぼ全要素non-zeroなので、CSR化によるindex領域の増加を避け、OneHotを含めて`float32`のdense結合を使います。高cardinalityカテゴリを大量に追加する場合は入力次元と表示されるメモリ警告を確認してください。

T1〜T3のLRは、行数より特徴数が多い高次元疎行列を想定して`liblinear, dual=True`を既定にしています。収束警告が出る場合は`CONFIG["tfidf_lr"]`の`max_iter`や`tol`を調整できます。

実行は`notebooks/04_embedding_models.ipynb`から行います。Notebookは初期状態で`RUN_CV=False`、`RUN_ENSEMBLE=False`、`RUN_FINAL_SUBMISSION=False`のため、明示的に有効化するまで重い学習やtest予測を開始しません。

TF-IDF設定と生成特徴量の保存先もconfigから変更できます。

```python
CONFIG["text_cols"] = ["project_name", "project_objective", "project_summary"]
CONFIG["tfidf"]["max_features"] = 300_000
CONFIG["tfidf_feature_output_dir"] = "data/csv/tfidf_shared"
```

共有TF-IDF特徴は`data/csv/tfidf_shared/fold_<n>_year_<year>/`へlong形式の圧縮CSVとして一度だけ保存します。T3はdenseなEmbeddingをCSRへ変換するため、`fold_metrics`へ`n_nonzero`と`sparse_memory_mib`を記録します。

前処理のimputer、scaler、OneHotEncoderとoptional PCAはfold trainingだけでfitします。OOFの古い年度は`NaN`のまま保持し、以下へ保存します。

```text
outputs/
├── oof_predictions.parquet
├── fold_metrics.parquet
├── experiment_summary.parquet
├── experiment_summary.csv
├── ensemble_weights.csv
├── ensemble_history.csv
├── ensemble_fold_scores.csv
├── ensemble_summary.csv
├── ensemble_oof.parquet
├── submission.csv
└── test_predictions/
```

## ROC-AUC hill climbing ensembleとsubmission

評価指標はROC-AUCです。`ensemble.py`は各候補が完全に同じOOF行を持つことを確認し、次の目的関数を選択できます。

- `pooled_auc`: 全validation年度を連結したAUC
- `mean_fold_auc`: 各fold AUCの単純平均
- `weighted_fold_auc`: 指定重みによる各fold AUCの加重平均

`blend_mode="rank"`では、各モデルのOOFをfold内でpercentile rank化してから混合します。testは各モデルについてtest全体でrank化し、学習済み重みを適用します。`probability`も同じNotebookで比較できます。validationにならなかった古い年度など、全モデル共通でOOFが`NaN`の行は探索から除外され、ensemble OOFでも`NaN`のままです。

```python
from ensemble import hill_climb_auc

ensemble_result = hill_climb_auc(
    suite.oof_predictions,
    train[TARGET_COL],
    folds=folds,
    objective="weighted_fold_auc",
    fold_weights=[0.2, 0.3, 0.5],  # 古いfold → 最新fold
    blend_mode="rank",
    max_steps=50,
)

display(ensemble_result.individual_scores.to_frame())
display(ensemble_result.weights[ensemble_result.weights > 0].to_frame())
print("weighted fold AUC:", ensemble_result.score)
print("pooled OOF AUC:", ensemble_result.pooled_auc)
```

Notebook 04は3目的関数×2 blend modeの6通りを同じ表へ出します。OOF上の最高値を自動採用すると選択バイアスが増えるため、既定では`SELECTED_ENSEMBLE="weighted_rank"`を明示し、比較表を見て人が変更する設計です。重みの個数が実際のfold数と違う場合はエラーにします。

Notebook 04の最終セルは、正のensemble weightを持つモデルだけを全trainで再fitし、test予測を同じ重みで合成します。`ID_COL`の値と行順をtestからそのまま保持し、`outputs/submission.csv`を次の2列で作成します。

```text
<ID_COL>,science_tech_decision
```

提出前にはNotebookが、IDの欠損・重複、予測件数、不正な確率、CSV再読込後の列と行順を検査します。hill climbingはOOFへの追加最適化なので、単体モデルより過学習しやすい点には注意し、各fold AUCや最新foldの傾向も併せて判断してください。

PCAなしが既定です。比較するときだけ変更します。

```python
CONFIG["e6_pca_dims"] = [None, 512, 256]
```

CatBoostのGPU学習は公式仕様上、同じseedでもbitwise deterministicではありません。厳密な再現性が必要な最終比較では`task_type="CPU"`も確認してください。
