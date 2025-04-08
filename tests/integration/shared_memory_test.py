import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# 必要なライブラリをインポート
try:
    import torch
    import numpy as np
    from multiprocessing import shared_memory
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドでインストールしてください:")
    print("poetry add torch numpy")
    sys.exit(1)

def test_shared_memory():
    """
    共有メモリの基本的な機能をテストする関数
    """
    print("共有メモリの基本テスト開始")
    
    # テスト用のデータ作成
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    size = data.nbytes
    
    print(f"テストデータ: {data}")
    print(f"データサイズ: {size} バイト")
    
    # 共有メモリ作成
    try:
        shm = shared_memory.SharedMemory(create=True, size=size, name="test_shm")
        print(f"共有メモリを作成しました: {shm.name}")
        
        # 共有メモリにデータをコピー
        shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        np.copyto(shm_array, data)
        print(f"データを共有メモリにコピーしました: {shm_array}")
        
        # 別のプロセスから同じ共有メモリにアクセスする代わりに、
        # ここでは同じプロセス内で新しいビューを作成してテスト
        existing_shm = shared_memory.SharedMemory(name="test_shm")
        existing_array = np.ndarray(data.shape, dtype=data.dtype, buffer=existing_shm.buf)
        print(f"共有メモリから読み込んだデータ: {existing_array}")
        
        # データを変更してみる
        existing_array[2] = 99.0
        print(f"変更後の共有メモリデータ: {shm_array}")
        
        # クリーンアップ
        existing_shm.close()
        shm.close()
        shm.unlink()
        print("共有メモリを解放しました")
        
        print("共有メモリの基本テスト完了")
        return True
    except Exception as e:
        print(f"共有メモリテスト中にエラーが発生: {e}")
        traceback.print_exc()
        try:
            shm.close()
            shm.unlink()
        except:
            pass
        return False

def test_with_freefem():
    """
    共有メモリを使用したPythonとFreeFEMの連携をテスト
    """
    print("\nPython-FreeFEM連携テスト開始")
    
    # MLaSO-d_shmファイルの存在を確認
    mlaso_file = Path("src/core/MLaSO-d_shm.py")
    if not mlaso_file.exists():
        print(f"MLaSO-d_shmファイルが見つかりません: {mlaso_file}")
        return False
    
    # importlibを使用して動的にモジュールをインポート
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlaso_module", mlaso_file)
        mlaso_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mlaso_module)
        MLaSO_d = mlaso_module.MLaSO_d
        print(f"MLaSO_dクラスを正常にインポートしました")
    except Exception as e:
        print(f"MLaSO-d_shmモジュールの読み込み中にエラーが発生: {e}")
        traceback.print_exc()
        return False
    
    # FreeFEMスクリプトファイルの存在確認
    freefem_script = Path("src/freefem/2Delastic-FEA-shm.edp")
    if not freefem_script.exists():
        print(f"FreeFEMスクリプトが見つかりません: {freefem_script}")
        return False
    
    print("必要なモジュールとファイルを確認しました")
    
    # FreeFEMコマンドの存在確認
    import subprocess
    try:
        result = subprocess.run(["FreeFem++", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"FreeFEM バージョン: {result.stdout.strip()}")
        else:
            print("FreeFEMコマンドがインストールされていますが、バージョンを取得できませんでした")
    except FileNotFoundError:
        print("FreeFEMコマンドが見つかりません。FreeFEMがインストールされていることを確認してください。")
        return False
    
    # テスト実行
    try:
        # MLaSO_dインスタンスを作成
        mlaso = MLaSO_d(verbose=True)
        print("MLaSO_dインスタンスを作成しました")
        
        # 密度場を初期化
        n_elements = 5883  # デフォルトの要素数
        mlaso.initialize_density(n_elements)
        print(f"密度場を初期化しました: 要素数 = {n_elements}")
        
        # 単一反復のテスト実行
        print("FreeFEMとの連携テスト実行 (単一反復)")
        try:
            success = mlaso.on_the_job_training(iteration=0)
            if success:
                print("FreeFEMとの連携テスト完了")
                print(f"結果サイズ: {mlaso.results.shape if mlaso.results is not None else 'None'}")
                if mlaso.results is not None:
                    compliance = mlaso.results[n_elements]
                    lamG = mlaso.results[n_elements + 1]
                    print(f"コンプライアンス: {compliance}")
                    print(f"ラグランジュ乗数: {lamG}")
            else:
                print("FreeFEMとの連携テストが失敗しました")
        except Exception as e:
            print(f"FreeFEMとの連携テスト中にエラーが発生: {e}")
            traceback.print_exc()
            success = False
        
        return success
        
    except Exception as e:
        print(f"テスト実行中にエラーが発生: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="共有メモリテスト")
    parser.add_argument("--basic", action="store_true", help="基本的な共有メモリ機能のみテスト")
    parser.add_argument("--freefem", action="store_true", help="FreeFEMとの連携をテスト")
    args = parser.parse_args()
    
    # 引数がない場合は両方のテストを実行
    if not (args.basic or args.freefem):
        args.basic = True
        args.freefem = True
    
    if args.basic:
        basic_success = test_shared_memory()
    else:
        basic_success = True
    
    if args.freefem and basic_success:
        freefem_success = test_with_freefem()
    else:
        freefem_success = False
    
    # 結果を表示
    print("\nテスト結果:")
    if args.basic:
        print(f"基本的な共有メモリテスト: {'成功' if basic_success else '失敗'}")
    if args.freefem:
        print(f"FreeFEMとの連携テスト: {'成功' if freefem_success else '失敗'}")
    
    # 終了コード設定
    sys.exit(0 if (basic_success and (not args.freefem or freefem_success)) else 1) 