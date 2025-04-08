#!/bin/bash
# WSL内でFreeFEMとPythonスクリプトを実行するためのシェルスクリプト

# パッケージのインストール（初回実行時のみ必要）
function install_packages() {
    echo "必要なパッケージをインストールしています..."
    sudo apt update
    sudo apt install -y python3-pip python3-numpy python3-torch
    pip3 install sysv-ipc
    
    # FreeFEMがインストールされているか確認
    if ! command -v FreeFem++ &> /dev/null; then
        echo "FreeFEMがインストールされていません。インストール方法については以下を参照してください："
        echo "https://doc.freefem.org/introduction/installation.html"
        exit 1
    fi
}

# テスト実行
function run_tests() {
    echo "共有メモリの基本テストを実行します..."
    python3 tests/integration/shared_memory_test.py --basic
    
    if [ $? -eq 0 ]; then
        echo "FreeFEMとの連携テストを実行します..."
        python3 tests/integration/shared_memory_test.py --freefem
    fi
}

# 最適化実行
function run_optimization() {
    echo "トポロジー最適化を実行します..."
    python3 src/core/MLaSO-d_shm.py --verbose --iterations 5
}

# メイン処理
case "$1" in
    "install")
        install_packages
        ;;
    "test")
        run_tests
        ;;
    "optimize")
        run_optimization
        ;;
    *)
        echo "使用方法: $0 [install|test|optimize]"
        echo "  install  - 必要なパッケージをインストールします"
        echo "  test     - 共有メモリとFreeFEMの連携テストを実行します"
        echo "  optimize - トポロジー最適化を実行します"
        ;;
esac

exit 0 