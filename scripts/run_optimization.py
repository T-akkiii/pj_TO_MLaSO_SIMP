#!/usr/bin/env python3
"""
トポロジー最適化実行スクリプト
トポロジー最適化アルゴリズムを実行するためのコマンドラインインターフェース
"""

import argparse
import sys
import os
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

def run_standard_optimization(iterations, verbose, config=None):
    """標準のトポロジー最適化を実行"""
    try:
        from src.core.MLaSO_d import MLaSO_d
        
        # インスタンス作成
        print(f"標準のトポロジー最適化を開始します（イテレーション: {iterations}）")
        mlaso = MLaSO_d(verbose=verbose)
        
        # 最適化実行
        mlaso.implementation(max_iterations=iterations)
        
        print("最適化が完了しました")
        return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

def run_shared_memory_optimization(iterations, verbose, config=None):
    """共有メモリを使用したトポロジー最適化を実行"""
    try:
        from src.core.MLaSO_d_shm import MLaSO_d
        
        # インスタンス作成
        print(f"共有メモリ版トポロジー最適化を開始します（イテレーション: {iterations}）")
        mlaso = MLaSO_d(verbose=verbose)
        
        # 最適化実行
        mlaso.implementation(max_iterations=iterations)
        
        print("最適化が完了しました")
        return True
    except ImportError:
        print("共有メモリモジュール（sysv_ipc）をインポートできません。")
        print("Linux環境で実行していることを確認してください。")
        print("また、 'pip install sysv-ipc' でモジュールをインストールしてください。")
        return False
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="トポロジー最適化実行スクリプト")
    parser.add_argument("--method", choices=["standard", "shared-memory"], default="standard",
                        help="使用する最適化方法: standard=標準, shared-memory=共有メモリ")
    parser.add_argument("--iterations", type=int, default=100,
                        help="最適化の反復回数")
    parser.add_argument("--verbose", action="store_true",
                        help="詳細な出力を有効にする")
    parser.add_argument("--config", type=str,
                        help="設定ファイルへのパス")
    
    args = parser.parse_args()
    
    if args.method == "standard":
        success = run_standard_optimization(args.iterations, args.verbose, args.config)
    elif args.method == "shared-memory":
        success = run_shared_memory_optimization(args.iterations, args.verbose, args.config)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 