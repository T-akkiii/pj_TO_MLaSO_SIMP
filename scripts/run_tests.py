#!/usr/bin/env python3
"""
テスト実行スクリプト
プロジェクトの各種テストを実行するスクリプト
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """単体テストの実行"""
    print("単体テストを実行します...")
    try:
        result = subprocess.run(["pytest", "-v", "tests/unit"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("pytestコマンドが見つかりません。")
        print("pip install pytestでインストールしてください。")
        return False

def run_integration_tests():
    """結合テストの実行"""
    print("結合テストを実行します...")
    try:
        result = subprocess.run(["pytest", "-v", "tests/integration"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("pytestコマンドが見つかりません。")
        print("pip install pytestでインストールしてください。")
        return False

def run_shared_memory_test():
    """共有メモリテストの実行"""
    print("共有メモリテストを実行します...")
    try:
        result = subprocess.run(["python", "tests/integration/shared_memory_test.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def run_to_test():
    """トポロジー最適化テストの実行"""
    print("トポロジー最適化テストを実行します...")
    try:
        result = subprocess.run(["python", "tests/integration/TO_test.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="テスト実行スクリプト")
    parser.add_argument("--unit", action="store_true", help="単体テストを実行")
    parser.add_argument("--integration", action="store_true", help="結合テストを実行")
    parser.add_argument("--shared-memory", action="store_true", help="共有メモリテストを実行")
    parser.add_argument("--to", action="store_true", help="トポロジー最適化テストを実行")
    parser.add_argument("--all", action="store_true", help="すべてのテストを実行")
    
    args = parser.parse_args()
    
    # デフォルトは全テスト実行
    if not (args.unit or args.integration or args.shared_memory or args.to):
        args.all = True
    
    if args.all:
        args.unit = args.integration = args.shared_memory = args.to = True
    
    success = True
    
    if args.unit:
        success = run_unit_tests() and success
    
    if args.integration:
        success = run_integration_tests() and success
    
    if args.shared_memory:
        success = run_shared_memory_test() and success
    
    if args.to:
        success = run_to_test() and success
    
    if success:
        print("\nすべてのテストが成功しました。")
    else:
        print("\n一部のテストが失敗しました。")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 