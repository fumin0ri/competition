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
│   ├── 04_embedding_models.ipynb
│   ├── 05_cohere_embeddings_to_submission.ipynb
│   ├── 06_tfidf_svd_nonlinear.ipynb
│   ├── 2_01_generate_ai_market_llm_features.ipynb
│   └── 2_02_analyze_ai_market_themes.ipynb
├── outputs/                    # モデルOOF・評価・test予測（Git管理対象外）
├── tests/
│   ├── test_validation.py
│   ├── test_text_features.py
│   ├── test_embedding_features.py
│   ├── test_modeling.py
│   ├── test_ensemble.py
│   ├── test_llm_features.py
│   └── test_market_analysis.py
├── requirements.txt
├── embedding_features.py      # Amazon Bedrock Embedding・resume・保存
├── ensemble.py                # AUC hill climbing・test blend・submission
├── llm_features.py            # 官公庁AI市場分析用LLM分類・validation・resume
├── market_analysis.py         # AIU重点テーマの集計・HHI・スコアリング
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
    text_cols=["project_name", "project_objective", "project_summary", "current_issues"],
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

## Amazon Bedrock Embedding

`embedding_features.py`はAmazon Bedrock Runtimeの`InvokeModel`を呼び出し、生成物を`data/embeddings/`へ保存します。認証にはboto3の標準credential chainを使用します。AWS上ではexecution role、ローカルではAWS profileなどを利用し、Access keyをNotebookや設定ファイルへ直接書きません。実行roleには対象modelへの`bedrock:InvokeModel`権限が必要です。

API引数は[AWS公式Boto3 InvokeModel](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/invoke_model.html)、既定payloadは[AWS公式Titan Text Embeddings request/response](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html)に合わせています。

```powershell
$env:AWS_DEFAULT_REGION="ap-northeast-1"
# ローカルでnamed profileを使う場合のみ
$env:AWS_PROFILE="your-profile"
```

最初に`notebooks/03_generate_embeddings.ipynb`のdry-runで、行数、文字数、推定token、truncate候補を確認します。dry-runではboto3 clientもBedrockも呼び出しません。少量の疎通確認は`RUN_SMOKE_API`、全件生成は`RUN_FULL_API`を明示的に`True`へ変更した場合だけ実行します。

```text
data/embeddings/
└── bedrock_amazon.titan-embed-text-v2_0_dim1024_<config-hash>/
    ├── config.json
    ├── train_embeddings.npy
    ├── train_metadata.parquet
    ├── train_progress.json
    ├── test_embeddings.npy
    ├── test_metadata.parquet
    ├── test_progress.json
    └── shards/
```

batchごとにshardを保存するため、中断後は取得済みbatchを再利用して続きから再開します。model ID、region、dimension、adapter ID、使用列、template、正規化、text hashが一致しないcacheは再利用しません。`EMBEDDING_DIM=None`なら最初のresponseから次元を解決します。

Bedrockはmodelごとに入出力仕様が異なるため、次の2つを差し替え可能にしています。

- `request_builder(text, model, embedding_dim)`: 1件のtextから`bytes`、文字列、またはJSON化可能なdictを返す
- `response_parser(body)`: response bodyから1次元`float32`互換vectorを返す

既定adapterはAmazon Titan Text Embeddings V2用です。`inputText`、`dimensions`、`normalize=true`を送り、`embedding`と`inputTextTokenCount`を読み取ります。Titan V2は1リクエスト1テキストなので、checkpoint batch内でも1件ずつ`InvokeModel`を呼び、リトライも1件単位です。model・payload・parserを変更したときは`adapter_id`も更新してください。

```python
from embedding_features import generate_embeddings

result = generate_embeddings(
    df=train,
    split="train",
    provider="bedrock",
    model="amazon.titan-embed-text-v2:0",
    region_name="ap-northeast-1",
    embedding_dim=1024,
    output_root="data/embeddings",
    dry_run=True,  # boto3 clientもBedrockも呼ばない
)
```

別のBedrock modelやimported modelが`sentence/result`形式の場合は、Notebook側だけで次のように変更できます。

```python
import json
import numpy as np

def request_builder(text, model, embedding_dim):
    return {"sentence": text, "parameters": {"dimension": embedding_dim}}

def response_parser(body):
    payload = json.loads(body.decode("utf-8"))
    vector = np.asarray(payload["result"], dtype=np.float32)
    assert vector.ndim == 1
    return vector

result = generate_embeddings(
    df=train,
    split="train",
    provider="bedrock",
    model="your-model-id-or-arn",
    request_builder=request_builder,
    response_parser=response_parser,
    adapter_id="sentence-result-v1",
    dry_run=False,
)
```

`performanceConfigLatency`、guardrail、service tierなどを渡す場合は`invoke_model_kwargs`を使用できます。kwargsを指定した場合は、秘密情報を含まない`bedrock_cache_identity`にも対応する識別情報を必ず入れてcache fingerprintへ反映します。料金はmodelとregionで異なるため、実行直前に現行の入力token単価を`price_per_million_tokens`へ明示設定します。

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
CONFIG["text_cols"] = ["project_name", "project_objective", "project_summary", "current_issues"]
CONFIG["tfidf"]["max_features"] = 300_000
CONFIG["tfidf_feature_output_dir"] = "data/csv/tfidf_shared"
```

共有TF-IDF特徴は`data/csv/tfidf_shared/fold_<n>_year_<year>/`へlong形式の圧縮CSVとして一度だけ保存します。T3はdenseなEmbeddingをCSRへ変換するため、`fold_metrics`へ`n_nonzero`と`sparse_memory_mib`を記録します。

前処理のimputer、scaler、OneHotEncoderとoptional PCAはfold trainingだけでfitします。OOFの古い年度は`NaN`のまま保持し、以下へ保存します。

tabularを使うE2・E4・E5・E6・T2・T3には、`CONFIG["text_cols"]`の各列からtarget非依存の文字統計も追加します。NFKC正規化後の欠損、文字数、`log1p`文字数、行数、文数、数字・英字・ひらがな・カタカナ・漢字・句読点の比率、unique文字率に加え、全列の合計文字数、非空列数、主要列間の長さ比を生成します。すべて同じ行の値だけから決まり、年度やtargetによる集計は行いません。

```python
CONFIG["feature_engineering"]["add_text_statistics"] = True  # 既定
```

無効化してablation比較する場合は`False`へ変更します。`project_name`頻度やtarget encodingのように複数行を使う特徴は、この処理には含めていません。

```text
outputs/
├── oof_predictions.parquet
├── fold_metrics.parquet
├── experiment_summary.parquet
├── experiment_summary.csv
├── ensemble_profile_comparison.csv
├── ensembles/
│   └── <profile>/             # profile別のweight・history・fold score・OOF
├── submission.csv
├── submissions/
│   ├── submission_uniform_rank.csv
│   ├── submission_recent_20_30_50_rank.csv
│   ├── submission_recent_10_20_70_rank.csv
│   ├── submission_recent_20_30_50_probability.csv
│   └── submission_manifest.csv
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

Notebook 04は、年度driftへの仮説を`ENSEMBLE_PROFILES`として複数定義します。既定では、全fold均等rank、最新年度を20/30/50または10/20/70で重視するrank、20/30/50のprobability blendを比較します。ここでのfold weightはtest年度を直接重み付けする値ではなく、hill climbingが各モデルの混合比を選ぶ際のfold AUCの重要度です。

```python
ENSEMBLE_PROFILES = {
    "uniform_rank": {
        "objective": "weighted_fold_auc",
        "blend_mode": "rank",
        "fold_weights": [1, 1, 1],
    },
    "recent_20_30_50_rank": {
        "objective": "weighted_fold_auc",
        "blend_mode": "rank",
        "fold_weights": [0.2, 0.3, 0.5],
    },
    "recent_10_20_70_rank": {
        "objective": "weighted_fold_auc",
        "blend_mode": "rank",
        "fold_weights": [0.1, 0.2, 0.7],
    },
    "recent_20_30_50_probability": {
        "objective": "weighted_fold_auc",
        "blend_mode": "probability",
        "fold_weights": [0.2, 0.3, 0.5],
    },
}
```

profileは自由に追加・削除できますが、`fold_weights`の個数は実際のfold数と一致させます。OOF上の最高値を自動採用すると選択バイアスが増えるため、基準profileは`SELECTED_ENSEMBLE_PROFILE`で明示します。

Notebook 04の最終セルは、全profileで正のensemble weightを持つモデルの和集合だけを全trainで一度ずつ再fitし、そのtest予測をprofile間で共有します。profile別の提出ファイルは`outputs/submissions/submission_<profile>.csv`、設定・CV値・モデル重みの一覧は`submission_manifest.csv`へ保存します。選択した基準profileは従来互換の`outputs/submission.csv`にも同内容で保存します。すべてのCSVは`ID_COL`の値と行順をtestからそのまま保持し、次の2列で作成します。

```text
<ID_COL>,science_tech_decision
```

提出前にはNotebookが、IDの欠損・重複、予測件数、不正な確率、CSV再読込後の列と行順を検査します。複数profileをLeaderboardへ出す場合も、結果を見て細かく重みを刻み続けるとPublic LBへ過適合します。まず仮説の異なる少数profileを比較し、`submission_manifest.csv`と提出スコアの対応を記録してください。

PCAなしが既定です。比較するときだけ変更します。

```python
CONFIG["e6_pca_dims"] = [None, 512, 256]
```

CatBoostのGPU学習は公式仕様上、同じseedでもbitwise deterministicではありません。厳密な再現性が必要な最終比較では`task_type="CPU"`も確認してください。

## Cohere MultilingualでEmbeddingから提出まで

`notebooks/05_cohere_embeddings_to_submission.ipynb`は、Amazon Bedrockの`cohere.embed-multilingual-v3`を使い、次を一続きで実行するNotebookです。

1. train/testのdry-runと料金概算
2. 5行のCohere API smoke test
3. resumableなtrain/test Embedding生成または既存cache読込
4. 同一時系列foldでE1〜E6 / T1〜T3を比較
5. Notebook 04のTitan OOFとCohere OOFを結合
6. 年度重みを変えたROC-AUC hill climbing
7. Titan/Cohereの選抜モデルを全train再fitして複数submission作成

Cohere Embed v3は分類用途として`input_type="classification"`、長文は`truncate="END"`を使います。出力は1024次元です。CohereのBedrock adapterは`embedding_features.py`に実装してあり、複数textを最大96件まで1 API callへまとめられます。Notebookの既定は32件/callです。単体テストでpayload、batch数、response shapeを検証しています。

安全のため、初期状態では次の実行フラグがすべて`False`です。dry-runと料金を確認してから、上から順に有効化してください。

```python
RUN_SMOKE_API = True
RUN_EMBEDDING_API = True
RUN_CV = True
RUN_ENSEMBLE = True
RUN_FINAL_SUBMISSION = True
```

既存Embeddingを再利用するときはAPIフラグを無効にし、`EXISTING_COHERE_CACHE_DIR`へcache directoryを指定します。Titanもensembleへ入れる既定設定では、先にNotebook 04を実行して`outputs/oof_predictions.parquet`を作成します。最終submission時にはNotebook 03で生成したTitan cacheを`TITAN_EMBEDDING_CACHE_DIR`へ指定してください。

統合時はEmbedding依存モデルを`cohere__E1_embedding_lr`、`titan__E1_embedding_lr`のように別候補にします。Embeddingを使わないE5/T1/T2は同じ特徴・モデルの重複を避けるため、Cohere側から一度だけ採用します。Cohere単独のCV成果物と、統合ensemble成果物は次のように分離されます。

```text
outputs/cohere/
├── oof_predictions.parquet
└── experiment_summary.csv

outputs/cohere_titan_ensemble/
├── oof_predictions.parquet
├── ensemble_profile_comparison.csv
├── ensembles/<profile>/
├── test_predictions/
├── submissions/
│   ├── submission_<profile>.csv
│   └── submission_manifest.csv
└── submission.csv
```

## 官公庁AI市場分析用LLM特徴量

`notebooks/2_01_generate_ai_market_llm_features.ipynb`は、train/testそれぞれの`project_start_year >= 2020`の行だけを同じ行政事業群として扱い、次の5列をAmazon Bedrockで生成します。年度は対象抽出と出力メタデータに使いますが、LLM本文には送りません。`-1`、欠損、解釈不能な年度、2019年以前は対象外です。

- `policy_domain`: D01〜D15から1件
- `admin_process`: A01〜A13から1件
- `ai_usecase`: U01〜U11またはU99から1〜3件
- `ai_applicability`: 0 / 1 / 2
- `classification_reason`: 人間による監査用の説明

AIU fit、市場性、期待効果、scalability等はLLMに出力させません。`llm_features.py`が余分な列を含むresponseも拒否し、U99と`ai_applicability=0`の整合性を含めて機械的にvalidationします。

既定候補は低コストのAmazon Nova Microです。AWS公式の[model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html)によるとConverseとtool useに対応しますが、厳密なStructured OutputsとTokyo in-regionには対応していません。そのため、強制tool callとローカルvalidationを併用し、Notebook既定リージョンは`us-east-1`です。リージョン要件がある場合は、利用可能な別モデルへ変更してください。モデル変更時は`MODEL_ID`だけでなく、`REGION_NAME`、入力・出力単価、model access、tool use/toolChoice対応、モデル固有の追加request fieldsも確認します。

初期状態では`RUN_SAMPLE_API=False`、`RUN_FULL_API=False`です。全件dry-runで件数・最大token・最大費用・sample promptを確認し、10件sampleの分類品質を人手確認してから全件処理します。単価は実行日に[AWS公式料金](https://aws.amazon.com/bedrock/pricing/)を再確認してください。

成功した行は1件ずつcheckpointされます。同じmodel、prompt version、入力列、推論設定で再実行すると成功済み行を再利用し、未処理・失敗行だけを呼び出します。見積り時と実行中の両方で`MAX_BUDGET_USD=20`を検査します。

```text
data/llm_features/bedrock_<model>_<prompt-version>_<config-hash>/
├── config.json
├── failures.jsonl
├── shards/projects_start_year_2020_plus/<row-position>.json
├── projects_start_year_2020_plus_features.csv.gz
├── projects_start_year_2020_plus_errors.csv.gz
└── projects_start_year_2020_plus_progress.json

data/csv/
└── ai_market_llm_features.csv.gz
```

canonical CSVには元の`project_id`、`source_split`、`source_index`、API用一意キー、分類結果、model、prompt version、token数、retry数を保存します。`ai_usecase`は`U02|U05`形式で、`decode_ai_usecase()`からlistへ戻せます。

## AIU重点テーマ分析

`notebooks/2_02_analyze_ai_market_themes.ipynb`は、Notebook 2_01の保存済み分類結果を読み込み、`admin_process × ai_usecase`を1テーマとして可視化・順位付けします。LLM APIは再度呼びません。`source_split + project_id`で元のtrain/testへone-to-one結合し、対象は`project_start_year >= 2020`だけです。

各テーマでは、High-app事業数・High-app率・省庁breadth・policy domain breadth・省庁/domainのHHI・明示的ルールによるAIU Fitを計算します。High-appは`ai_applicability == 2`だけを指します。生のHigh-app率と、全体率を事前値・強度20件として平滑化した順位用の率を分けて保存します。複数ラベルの`ai_usecase`はexplodeし、`U99`とHigh-app 0件のテーマはランキング対象外です。

初期の総合重みは、High-app件数20%、平滑化率15%、省庁breadth15%、domain breadth10%、低集中度10%、AIU Fit30%です。Notebook上部の辞書・重みを編集でき、戦略重視・均等・市場規模重視の順位感度も比較します。

```text
data/csv/
├── ai_market_theme_metrics.csv
└── ai_market_theme_yearly_metrics.csv

outputs/ai_market_themes/
├── top_themes.csv
├── weight_profile_comparison.csv
└── *.png
```

入力CSVと`data/csv/ai_market_llm_features.csv.gz`を配置した後、Notebookを上から実行してください。結合漏れ、重複キー、年度不一致、未知taxonomyがある場合は分析を停止します。

## TF-IDF SVD + MLP/XGBoost

`notebooks/06_tfidf_svd_nonlinear.ipynb`は、保存済みのTitanまたはCohere Embeddingを選び、次の2実験を既存と同じ時系列foldで評価します。

| ID | 入力 | モデル |
|---|---|---|
| S1 | char TF-IDF → TruncatedSVD + Embedding + tabular/text統計 | MLP |
| S2 | char TF-IDF → TruncatedSVD + Embedding + tabular/text統計 | XGBoost |

raw TF-IDFはCSRのまま保持し、各foldのtrainingだけでfitしたTruncatedSVDによって既定256次元へ圧縮します。TF-IDFとSVDはS1/S2で共有し、validationやtestはtransformだけに使います。MLPはAUC early stopping、XGBoostは`eval_metric="auc"`を使用します。

```python
CONFIG["tfidf_svd"]["n_components"] = 256
RUN_CV = True
RUN_FULL_TEST_PREDICTION = True  # CV確認後のみ
```

fold別およびfull-trainのSVD特徴は`data/csv/tfidf_svd_shared/`、OOF・metrics・個別test予測・submissionは`outputs/tfidf_svd_nonlinear/`へ保存します。指定次元を確保できないfoldでは自動縮小せず、設定変更を促すエラーにします。
