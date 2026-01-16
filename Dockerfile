# 1. NVIDIAが提供する最新のPyTorch専用イメージ（CUDA 12.8 / PyTorch 2.5相当）
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 2. 環境変数の設定(対話プロンプト防止)
ENV DEBIAN_FRONTEND=noninteractive

# 3.システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 作業ディレクトリの設定
WORKDIR /workspace

# 5. requirements.txt をコピー
COPY requirements.txt .

# 6. 【一括インストール】
# -r を使って requirements.txt を読み込みます
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 起動時はbashを立ち上げる
CMD ["bash"]