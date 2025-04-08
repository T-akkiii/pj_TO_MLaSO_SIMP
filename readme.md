# トポロジー最適化プロジェクト

このプロジェクトは、FreeFEMとPythonを連携させたトポロジー最適化フレームワークを提供します。機械学習支援型形状最適化（MLaSO）手法を実装し、共有メモリを使用した効率的なデータ通信機能も含んでいます。

## 機能

- **機械学習支援トポロジー最適化** - ディープラーニングモデルを用いて感度解析を補助
- **共有メモリ連携** - FreeFEMとPython間でデータを効率的に受け渡し（Linux環境用）
- **レベルセット法** - レベルセット法によるトポロジー最適化の実装
- **テスト機能** - 各コンポーネントのテスト実装

## ディレクトリ構造

```
├── src/                 # ソースコード
│   ├── core/            # コア機能
│   │   ├── MLaSO-d.py   # 基本的なML支援形状最適化アルゴリズム
│   │   └── MLaSO-d_shm.py # 共有メモリ版ML支援形状最適化アルゴリズム
│   │
│   ├── freefem/         # FreeFEMスクリプト
│   │   ├── 2Delastic-FEA.edp        # 基本的な有限要素解析スクリプト
│   │   ├── 2Delastic-FEA-shm.edp    # 共有メモリ版有限要素解析スクリプト
│   │   ├── 2Delastic-FEA-test.edp   # テスト用有限要素解析スクリプト
│   │   ├── 2Delastic-geom.edp       # ジオメトリ関連スクリプト
│   │   ├── sensitivity-analysis.edp # 感度解析スクリプト
│   │   ├── set-mesh-and-bc.edp      # メッシュと境界条件設定スクリプト
│   │   ├── save_vtk.edp             # VTKファイル保存スクリプト
│   │   └── reaction-difusion-equation.edp # 反応拡散方程式スクリプト
│   │
│   ├── utils/           # ユーティリティスクリプト
│   │   ├── sensitivity_net.py      # 感度ネットワーク関連スクリプト
│   │   └── wsl_run.sh              # WSL実行用スクリプト
│   │
│   ├── mesh/            # メッシュファイルを含むディレクトリ
│   ├── SaveData/        # データ保存ディレクトリ
│   └── SaveVTK/         # VTK形式データ保存ディレクトリ
│
├── scripts/             # 実行スクリプト
│   ├── run_optimization.py  # 最適化実行スクリプト
│   └── run_tests.py         # テスト実行スクリプト
│
├── tests/               # テストコード
│   ├── unit/            # 単体テスト
│   └── integration/     # 結合テスト
│       ├── shared_memory_test.py   # 共有メモリテスト
│       ├── TO_test.py              # トポロジー最適化テスト
│       ├── test_pyfreefem.py       # Python-FreeFEM連携テスト
│       ├── levset-TO-test.py       # レベルセットベーストポロジー最適化テスト
│       └── schedule_free_test.py   # スケジュールテスト
│
├── config/              # 設定ファイル
├── notebooks/           # Jupyter notebooks
├── data/                # データファイル
└── document/            # ドキュメント
```

## 主要機能の説明

### 1. 通常のトポロジー最適化 (`src/core/MLaSO-d.py`)
- ファイルベースのデータ通信を使用
- FreeFEMスクリプト `src/freefem/2Delastic-FEA.edp` を呼び出して有限要素解析を実行

### 2. 共有メモリを使用したトポロジー最適化 (`src/core/MLaSO-d_shm.py`)
- Linux環境での共有メモリを使用したデータ通信
- FreeFEMスクリプト `src/freefem/2Delastic-FEA-shm.edp` を呼び出して有限要素解析を実行
- 一時ファイルの読み書きを回避し、処理速度を向上

### 3. レベルセット法によるトポロジー最適化
- レベルセット関数を使用した境界表現
- 反応拡散方程式による形状更新

## 必要条件

- Python 3.x
- FreeFEM 4.x（共有メモリ機能を使用する場合は`mmap-semaphore`プラグイン付き）
- PyTorch
- Linux環境（共有メモリ機能を使用する場合）
- WSL（Windows環境で共有メモリ機能を使用する場合）

## インストール

WSL環境では以下のコマンドで必要なパッケージをインストールできます：

```bash
bash src/utils/wsl_run.sh install
```

または Poetry を使用して依存関係をインストール：

```bash
poetry install
```

FreeFEMが必要です。インストール方法は[公式ドキュメント](https://doc.freefem.org/introduction/installation.html)を参照してください。

## 使用方法

### スクリプトによる実行

```bash
# 標準のトポロジー最適化
python scripts/run_optimization.py --method standard --iterations 100

# 共有メモリ版トポロジー最適化
python scripts/run_optimization.py --method shared-memory --iterations 100

# テスト実行
python scripts/run_tests.py --all
```

### コア関数の直接実行

```bash
# 標準のトポロジー最適化
poetry run python src/core/MLaSO-d.py

# 共有メモリ版トポロジー最適化 (Linux/WSL環境)
bash src/utils/wsl_run.sh optimize
```

### テスト実行

```bash
# WSLを使用したテスト
bash src/utils/wsl_run.sh test

# pytestによるテスト
poetry run pytest tests/

# 特定のテスト実行
python tests/integration/shared_memory_test.py
```

## 共有メモリの動作原理

共有メモリを使用したPythonとFreeFEMの連携は以下のように機能します：

1. Pythonで共有メモリセグメントを作成
2. 密度場とパラメータを共有メモリに書き込む
3. 準備完了フラグファイルを作成
4. FreeFEMプロセスを起動し、共有メモリ名を引数として渡す
5. FreeFEMが共有メモリからデータを読み込む
6. FreeFEMが計算を実行し、結果を共有メモリに書き込む
7. 完了フラグファイルを作成
8. Pythonが完了フラグを検出し、共有メモリから結果を読み込む

## 注意事項

- 共有メモリ機能はLinux環境（WSL含む）でのみ動作します。Windows環境では通常のファイルベース通信を使用してください。
- FreeFEMがインストールされていることを確認してください。
- Linux環境で共有メモリを使用する場合は、`sysv-ipc` Pythonライブラリが必要です。
- WSLを使用する場合は、ファイルパスの互換性に注意してください。