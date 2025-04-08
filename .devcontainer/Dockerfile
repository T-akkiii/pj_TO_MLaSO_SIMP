FROM python:3.10-slim

# システムのアップデートと基本パッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    software-properties-common \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 開発ツールのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 基本的な開発環境の依存関係をインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    gfortran \
    mpich \
    libmpich-dev \
    m4 \
    automake \
    autoconf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# FreeFem++のソースコードをダウンロードしてビルド
RUN cd /tmp && \
    wget -q https://github.com/FreeFem/FreeFem-sources/archive/refs/tags/v4.12.tar.gz && \
    tar xzf v4.12.tar.gz && \
    cd FreeFem-sources-4.12 && \
    autoreconf -i && \
    ./configure --enable-download --enable-optim && \
    make -j4 && \
    make install && \
    cd / && \
    rm -rf /tmp/FreeFem-sources-4.12 /tmp/v4.12.tar.gz

# Poetry をインストール
ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.7.1
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# ワークディレクトリを設定
WORKDIR /workspace

# プロジェクトの依存関係をインストール
COPY pyproject.toml poetry.lock* /workspace/
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# ユーザーを作成
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ユーザーを切り替え
USER $USERNAME