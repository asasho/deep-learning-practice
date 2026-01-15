# 機械学習プロジェクト

このプロジェクトは、PyTorchを使用した機械学習の学習・実験プロジェクトです。
Cursorのdev containerを使用して開発環境を構築します。

## 必要な環境

- Docker
- Cursor（VS Codeベースのエディタ）

## セットアップ手順

### 0. プロジェクトディレクトリへの移動

```bash
cd ~/Desktop/deep_learning
```

### 1. devcontainer.jsonの作成（初回のみ）

`.devcontainer/devcontainer.json`ファイルを作成します：

```bash
mkdir -p .devcontainer
```

`.devcontainer/devcontainer.json`の内容：

```json
{
  "name": "Deep Learning Dev Container",
  "build": {
    "context": "..",
    "dockerfile": "../Dockerfile"
  },
  "runArgs": [
    "--gpus=all",
    "--shm-size=16g"
  ],
  "workspaceFolder": "/workspace",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3"
      }
    }
  }
}
```

### 2. Cursorでdev containerを開く

1. Cursorでプロジェクトフォルダを開く
2. コマンドパレット（`Cmd+Shift+P` / `Ctrl+Shift+P`）を開く
3. 「Dev Containers: Reopen in Container」を選択
4. 初回はイメージのビルドが開始されます（数分かかることがあります）

これで、Cursor内でdev containerが起動し、開発を開始できます。

## 開発手順

### 1. プロジェクトディレクトリへの移動

```bash
cd ~/Desktop/deep_learning
```

### 2. プロジェクトのクローン（初回のみ、リポジトリから取得する場合）

```bash
git clone <repository-url>
cd <project-directory>
```

### 3. Cursorでdev containerを開く

Cursorでプロジェクトを開き、dev containerで起動します（上記「セットアップ手順」を参照）。

### 4. 依存関係の確認

dev containerを起動すると、`requirements.txt`に記載されているパッケージが自動的にインストールされています。

追加のパッケージをインストールする場合：

1. requirements.txtにパッケージ名を追記
2. dev containerを再ビルド
    a. コマンドパレットを開く
    b. 「Dev Containers: Rebuild Container」を選択



### 5. プロジェクトの構成

- `calculation_graph/`: 計算グラフ関連の実装
- `error_function/`: 損失関数の実装
- `feed_forward_network/`: 順伝播ネットワーク
- `logistic_regression/`: ロジスティック回帰
- `k_means/`: K-meansクラスタリング
- `normalization/`: 正規化の実装
- `principal_component_analysis/`: 主成分分析
- `support_vector_machine/`: サポートベクターマシン

### 6. コードの編集

Cursorのエディタで各ディレクトリ内のPythonファイルやJupyter Notebookを編集して実装を進めます。
Jupyter NotebookはCursor内で直接実行・デバッグできます。

## 便利コマンド

### Cursor / Dev Container関連

#### dev containerの再ビルド

dev containerの設定を変更した場合や、Dockerfileを更新した場合：

1. コマンドパレット（`Cmd+Shift+P` / `Ctrl+Shift+P`）を開く
2. 「Dev Containers: Rebuild Container」を選択

#### dev containerの再起動

1. コマンドパレットを開く
2. 「Dev Containers: Reopen Folder Locally」でローカルに戻る
3. 再度「Dev Containers: Reopen in Container」でコンテナに戻る

または、ターミナルから：

```bash
# コンテナ内のターミナルで
exit
```

その後、Cursorで再度dev containerを開く

### Docker関連

#### Dockerイメージのビルド

```bash
docker build -t <イメージ名> .
```

#### コンテナの起動（インタラクティブ）

```bash
docker run --gpus all -it --rm \
  --name deep-learning \
  -v "$PWD":/workspace \
  --workdir /workspace \
  --shm-size=16g \
  <イメージ名:タグ名>
```

#### コンテナの起動（バックグラウンド）

```bash
docker run --rm --gpus all -d \
  --name deep-learning \
  -v "$PWD":/workspace \
  --workdir /workspace \
  --shm-size=16g \
  <イメージ名:タグ名> \
  tail -f /dev/null
```

#### 実行中のコンテナに入る

```bash
docker exec -it <コンテナ名> bash
```

#### 実行中のコンテナの確認

```bash
docker ps
```

#### コンテナのログ確認

```bash
docker logs -f <コンテナ名>
```

#### コンテナの停止

```bash
docker stop <コンテナ名>
```

#### コンテナの削除

```bash
docker rm <コンテナ名>
```

#### コンテナの停止と削除を同時に実行

```bash
docker rm -f <コンテナ名>
```

### Python関連

#### パッケージのインストール

```bash
pip install <package-name>
```

#### パッケージのアンインストール

```bash
pip uninstall <package-name>
```

#### インストール済みパッケージの確認

```bash
pip list
```

#### 依存関係の更新

```bash
pip freeze > requirements.txt
```

## プロジェクト構造

```
.
├── .devcontainer/              # Dev Container設定
│   └── devcontainer.json       # Dev Container設定ファイル
├── Dockerfile                  # Dockerイメージの定義
├── requirements.txt            # Python依存パッケージ一覧
├── README.md                   # このファイル
├── calculation_graph/          # 計算グラフ関連
│   ├── activations.py
│   ├── losses.py
│   ├── model.py
│   ├── optimizer.py
│   └── train.py
├── error_function/             # 損失関数
│   ├── binary_cross_entropy.ipynb
│   ├── cross_entropy_errors.ipynb
│   └── sum_squared_errors.ipynb
├── feed_forward_network/       # 順伝播ネットワーク
│   ├── common/
│   │   └── functions.py
│   ├── dataset/
│   └── feed_forward_network.ipynb
├── e_learning.ipynb            # 基本的な学習ノートブック
├── JDLA_E_PyTorch.ipynb        # JDLA試験関連
├── k_means/                    # K-meansクラスタリング
├── logistic_regression/        # ロジスティック回帰
├── normalization/              # 正規化
├── principal_component_analysis/  # 主成分分析
└── support_vector_machine/     # サポートベクターマシン
```

## トラブルシューティング

### GPUが認識されない場合

dev containerでGPUが認識されない場合、`.devcontainer/devcontainer.json`の`runArgs`を確認してください：

```json
"runArgs": [
  "--gpus=all",
  "--shm-size=16g"
]
```

それでも動作しない場合は、以下のように変更してください：

```json
"runArgs": [
  "--runtime=nvidia",
  "-e", "NVIDIA_VISIBLE_DEVICES=all",
  "--shm-size=16g"
]
```

### dev containerが起動しない場合

1. Dockerが起動しているか確認
2. コマンドパレットから「Dev Containers: Show Container Log」でログを確認
3. `.devcontainer/devcontainer.json`の構文エラーがないか確認

### 権限エラーが発生する場合

コンテナ内でファイルの作成・編集時に権限エラーが発生する場合は、ホスト側でファイルの権限を確認してください。
1. コマンドパレットを開く
2. 「Dev Containers: Rebuild Container」を選択
### パッケージがインストールされていない場合

dev containerを再ビルドしてください：

1. コマンドパレットを開く
2. 「Dev Containers: Rebuild Container」を選択

## 参考リンク

- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/index.html)
- [Cursor公式サイト](https://cursor.sh/)
- [Dev Containers公式ドキュメント](https://containers.dev/)
- [NVIDIA PyTorch Dockerイメージ](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

