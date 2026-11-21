# 各モジュールの入出力全体を学習する

`moduleio` は、PyBot の各モジュールを教師として、その呼び出しの入出力全体を
公式 Griffith で学習する実験です。入力文を既存の解析処理で6個の条件フラグに
変換してから判断させる方式ではなく、元の呼び出し引数を入力にします。

**初回の未学習18件の検証では、論理モデル・FP対照ともに全入出力一致は0件でした。**
全入出力を教師にする仕組みを実装した段階であり、元モジュールの置換に使える
学習済みモデルが完成したという結果ではありません。

例えば `respond(text, memory)` については、次の関数を学習対象にします。

```python
(module_id, text, memory_before, seed, date)
    -> (return_value, memory_after, exception)
```

返答だけ合っていても、更新後のメモリが違えば「全体一致」には数えません。
`None` を返した場合も、メモリ更新や例外を含めて比較します。

## 実装した境界

| 呼び出し口 | 入力 | 教師出力 |
|---|---|---|
| `observe` | 入力文・呼び出し前メモリ | 戻り値・呼び出し後メモリ・例外 |
| `respond` | 入力文・呼び出し前メモリ | 回答候補または `None`・呼び出し後メモリ・例外 |
| `choose` | 入力文・メモリ・候補一覧 | 選択結果・呼び出し後メモリ・呼び出し後候補一覧・例外 |
| `after` | 入力文・確定した回答・メモリ | 戻り値・呼び出し後メモリ・例外 |

元 PyBot の対象リビジョン `7d63c7fbebb44c7e466e5c8b5c1f341ab767929e` には、
これらの呼び出し口を持つモジュールが102個、呼び出し口が合計116個あります。
内訳は observe 16、respond 97、choose 1、after 2 です。

会話を記録するときは元の runtime の呼び出しを一時的に包みます。observe の
順次更新、respond ごとのメモリコピー、採用された候補のメモリだけを引き継ぐ
処理をそのまま実行します。採用されなかった候補の入出力も収集します。
元の `bot/runtime.py` は変更していません。

通常のJSONではtupleがlistに変わるため、型付きのシリアライズを用います。
辞書の順序、キーの型、大きな整数、浮動小数のビット、bytes、共有参照も保存します。
循環参照や独自クラスは、黙って文字列に変換せず明示的にエラーにします。

## モデルが行う処理

元の引数全体をUTF-8バイト列にし、モジュールIDとともに入力します。学習するのは、
完全な出力を型付きJSONで書き出したバイト列と、真の終端記号EOSです。
入力部分に損失はかけず、出力全体に損失をかけます。

使用する学習器は公式 Griffith の `LogicLadderGPT` です。LUT、緩和、代理勾配、
hard化の処理を独自に再実装していません。ライブラリの固定コミットは
`e0675f4e36596954aa424d522e0ee966bba9a3e1` です。

論理投影と論理FFNを使いますが、埋め込み・正規化・attentionのスコアとsoftmax・
出力headには浮動小数点演算が残ります。これは「モデル全体を純粋な論理回路に
変換した」という実装ではありません。hard評価はLUTを確定させた経路の評価です。

推論時は学習済み重みと入力だけを使います。元モジュールを呼び出したり、正解の
回答候補を検索したり、不正出力を元のルールで補ったりする経路はありません。
未学習の呼び出し口、長さ超過、EOS未生成、不正な出力は明示的に失敗します。

## 教師データの2つの収集方法

### 独立した呼び出し

`capture_call` は、指定した入力メモリをコピーし、指定seedで1回の元モジュールを
実行します。呼び出し後には外側の乱数状態を戻します。今回の小規模学習はこの
契約を使います。入力ファイルには質問とメモリだけを用意し、回答は元モジュールに
実行させて取得します。

```sh
python examples/moduleio/make_specs.py
python -m moduleio collect-calls \
  --spec examples/moduleio/calls.jsonl \
  --output examples/moduleio/teacher_calls.jsonl.gz
```

120件のサンプルは4モジュール・6呼び出し口を扱います。

| モジュール | 呼び出し口 | 題材 |
|---|---|---|
| `favourites` | observe / respond | 好きな色の記録と回答 |
| `group_ages` | observe / respond | 複数人の年齢の記録と回答 |
| `name_taken_back` | observe | 間違った名前の取り消しと状態更新 |
| `a_cake_between_its_layers` | respond | 長い回答文全体、該当しない質問 |

同じ色・年齢・名前の言い換えは同じgroupにまとめます。ケーキについては固定の
長い回答を持つ1つのfamilyとして扱います。split seed 1では訓練102件、検証18件です。
単一familyしかない呼び出し口は訓練側に保持し、検証対象に含まれないことを記録します。

### 実際の会話内の全呼び出し

```sh
python -m moduleio collect \
  --scenarios examples/moduleio/conversations.jsonl \
  --output examples/moduleio/runtime_trace.jsonl.gz
```

このトレースは実際の会話で流れるメモリと乱数の進行を保持します。2ターンから
232件、116種類すべての呼び出し口を記録したサンプルを同梱します。
各行には乱数状態の前後も欠落なく保存します。モデル用入力へ変換するときは
呼び出し前の乱数状態を入力に含めます。

このため自然会話のトレースは小型モデルのコンテキストを大きく超えることがあります。
訓練では上限を超えたレコードを明示的に除外し、件数と理由をレポートします。
入力や出力を無言で切り詰める処理はありません。検証時の長さ超過は失敗に数えます。

## 学習と検証

Python 3.11以上で `pip install '.[moduleio]'` を実行してください。

```sh
python -m moduleio train \
  --data examples/moduleio/teacher_calls.jsonl.gz \
  --checkpoint experiments/moduleio/model.pt \
  --report experiments/moduleio/training.json \
  --max-seq-len 1280 --d-model 64 --n-layers 1 \
  --steps 1000 --batch-size 4 --learning-rate 0.003 \
  --train-mode ste --seed 0 --split-seed 1 --threads 1

python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/model.pt.validation.jsonl.gz \
  --output experiments/moduleio/hard-validation.json --mode hard
```

`--train-mode ste` は公式の hard straight-through 経路を使います。
`--mode soft` で決定的なsoft自由生成と比較できます。

主指標は、正解出力を一切入力せずに自由生成した結果の **whole_call_exact** です。
次を個別にも報告します。

- 戻り値、メモリ、候補、例外それぞれの完全一致。
- 完全で有効な型付きJSONを生成し、EOSまで到達した割合。
- 回答または状態変化があるactiveケースと、無処理のinactiveケース。
- モジュール別・呼び出し口別の結果。
- 常に `None`、メモリ無変更と答える基準の結果。
- 長さ超過、不正JSON、型不整合、EOS未生成の件数。

学習レポートの teacher-forced token accuracy は、正解の接頭辞を与えた次バイト予測です。
これが高くても、モジュールの全入出力を再現できたとは数えません。

## 推論API

```python
from moduleio.predictor import load_predictor

predictor = load_predictor("experiments/moduleio/model.pt")
result = predictor.predict_call(
    "group_ages", "respond",
    {"text": "how old are they",
     "memory": {"group_ages": {
         "word": "kids", "ages": [4, 7], "line": "my kids are 4 and 7"}}},
    seed=0,
    environment_date="2026-11-21",
)
# result: {"return": ..., "memory": ..., "exception": ...}
```

このAPIは更新後メモリを返します。実際の会話メモリへの自動適用や、未学習の
元モジュールへのフォールバックは行いません。不正・不完全な予測は
`PredictionError` になります。提供重みの成功範囲は測定結果を参照してください。

自然会話トレースで学習したモデルには、`rng_state=` で記録された呼び出し前の
乱数状態を渡します。checkpointの `training_fixtures` と異なる入力方式は拒否します。
CLIでは `--rng-state` で型付きJSONを指定でき、tupleなどを含む引数ファイルは
`--typed-arguments` で読み込めます。

## 達成範囲と限界

この実験でいう全入出力は、4種類の明示的なhook引数・戻り値・引数の状態変化・
例外です。Pythonプロセスのあらゆる副作用まで等価にするという意味ではありません。
日付は記録しますが時計を仮想化しません。モジュールのグローバルキャッシュ全体や、
外側の乱数生成器の更新をモデルで再現する保証もありません。独立呼び出しは指定seed、
指定した元コード、記録日における実行を教師とする限定された実験です。

全116呼び出し口を収集できること、6呼び出し口に学習を実行したこと、それらが
未知入力でも正しく動くことは別々に評価します。有限の学習・検証例だけから、
任意の文章長・メモリ・数値について元プログラムと完全等価だとは主張しません。
また、今回の検証は元の入力メモリを与える1呼び出し評価です。予測したメモリを
繰り返し引き継ぐ会話全体の成功率ではありません。

実測値・失敗例・実行条件は [RESULTS.md](../../experiments/moduleio/RESULTS.md) と
隣接するJSONレポートに記載します。
