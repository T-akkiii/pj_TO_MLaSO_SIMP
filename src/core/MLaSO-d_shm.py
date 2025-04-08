#!/usr/bin/env python3
"""
MLaSO-d_shm.py - ML-assisted Shape Optimization - Deterministic.
共有メモリを使用してFreeFEMと効率的に通信するバージョン
"""

import os
import sys
import time
import tempfile
import argparse
import subprocess
import numpy as np
import sysv_ipc  # Linux環境での共有メモリライブラリ

class SharedMemoryLinux:
    """
    Linux環境での共有メモリ管理クラス
    FreeFEMのmmap-semaphoreプラグインと互換性があります
    """
    def __init__(self, verbose=False):
        self.shared_memories = {}
        self.semaphores = {}
        self.verbose = verbose
        self._temp_dir = None
        self._ready_flag_file = None
        self._done_flag_file = None
        
    def _get_temp_dir(self):
        """一時ディレクトリを取得"""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="mlaso_shm_")
            if self.verbose:
                print(f"一時ディレクトリを作成しました: {self._temp_dir}")
        return self._temp_dir
        
    def create_flag_files(self):
        """フラグファイルのパスを作成"""
        temp_dir = self._get_temp_dir()
        self._ready_flag_file = os.path.join(temp_dir, "ready_flag")
        self._done_flag_file = os.path.join(temp_dir, "done_flag")
        
        # done_flagが存在したら削除
        if os.path.exists(self._done_flag_file):
            os.remove(self._done_flag_file)
            
        return self._ready_flag_file, self._done_flag_file
        
    def create_shared_memory(self, name, size_bytes):
        """共有メモリセグメントを作成"""
        if self.verbose:
            print(f"共有メモリを作成: {name}, サイズ: {size_bytes}バイト")
            
        # キーを生成（名前から整数値に変換）
        key = abs(hash(name)) % 1000000
        
        try:
            # 既存の共有メモリがあれば削除
            try:
                existing_memory = sysv_ipc.SharedMemory(key)
                existing_memory.remove()
                if self.verbose:
                    print(f"既存の共有メモリを削除: {name}")
            except:
                pass
                
            # 新しい共有メモリを作成
            memory = sysv_ipc.SharedMemory(key, sysv_ipc.IPC_CREAT, size=size_bytes)
            
            # セマフォも作成
            try:
                existing_semaphore = sysv_ipc.Semaphore(key)
                existing_semaphore.remove()
                if self.verbose:
                    print(f"既存のセマフォを削除: {name}")
            except:
                pass
                
            semaphore = sysv_ipc.Semaphore(key, sysv_ipc.IPC_CREAT)
            
            self.shared_memories[name] = memory
            self.semaphores[name] = semaphore
            
            if self.verbose:
                print(f"共有メモリを作成しました: {name}, キー: {key}")
                
            return name
            
        except Exception as e:
            print(f"共有メモリ作成エラー ({name}): {str(e)}")
            return None
            
    def write_to_shared_memory(self, name, data):
        """共有メモリにデータを書き込む"""
        if name not in self.shared_memories:
            print(f"エラー: 共有メモリが見つかりません: {name}")
            return False
            
        memory = self.shared_memories[name]
        
        try:
            # NumPy配列をバイト列に変換
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            else:
                # 単一の値または標準のPythonリスト/配列
                data_array = np.array(data, dtype=np.float64)
                data_bytes = data_array.tobytes()
                
            # 共有メモリに書き込み
            memory.write(data_bytes)
            
            if self.verbose:
                print(f"共有メモリに書き込みました: {name}")
                if isinstance(data, np.ndarray):
                    print(f"  サイズ: {data.size}, 形状: {data.shape}")
                    if data.size < 10:
                        print(f"  データ: {data}")
                    else:
                        print(f"  データ (最初の5要素): {data[:5]}")
                        
            return True
            
        except Exception as e:
            print(f"共有メモリ書き込みエラー ({name}): {str(e)}")
            return False
            
    def read_from_shared_memory(self, name, shape=None, dtype=np.float64):
        """共有メモリからデータを読み込む"""
        if name not in self.shared_memories:
            print(f"エラー: 共有メモリが見つかりません: {name}")
            return None
            
        memory = self.shared_memories[name]
        
        try:
            # 共有メモリから読み込み
            data_bytes = memory.read()
            
            # バイト列をNumPy配列に変換
            if shape is not None:
                data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
            else:
                data = np.frombuffer(data_bytes, dtype=dtype)
                
            if self.verbose:
                print(f"共有メモリから読み込みました: {name}")
                if data.size < 10:
                    print(f"  データ: {data}")
                else:
                    print(f"  データ (最初の5要素): {data[:5]}")
                    
            return data
            
        except Exception as e:
            print(f"共有メモリ読み込みエラー ({name}): {str(e)}")
            return None
            
    def create_ready_flag(self):
        """準備完了フラグファイルを作成"""
        with open(self._ready_flag_file, 'w') as f:
            f.write("ready")
        if self.verbose:
            print(f"準備完了フラグファイルを作成しました: {self._ready_flag_file}")
            
    def wait_for_done_flag(self, timeout=60):
        """完了フラグファイルを待機"""
        start_time = time.time()
        while not os.path.exists(self._done_flag_file):
            if time.time() - start_time > timeout:
                print(f"タイムアウト: 完了フラグファイルが作成されませんでした: {self._done_flag_file}")
                return False
            time.sleep(0.1)
            
        if self.verbose:
            print(f"完了フラグファイルを検出しました: {self._done_flag_file}")
        return True
        
    def cleanup(self):
        """共有メモリとフラグファイルをクリーンアップ"""
        # 共有メモリを解放
        for name, memory in self.shared_memories.items():
            try:
                memory.remove()
                if self.verbose:
                    print(f"共有メモリを解放しました: {name}")
            except:
                pass
                
        # セマフォを解放
        for name, semaphore in self.semaphores.items():
            try:
                semaphore.remove()
                if self.verbose:
                    print(f"セマフォを解放しました: {name}")
            except:
                pass
                
        # フラグファイルを削除
        if self._ready_flag_file and os.path.exists(self._ready_flag_file):
            os.remove(self._ready_flag_file)
            
        if self._done_flag_file and os.path.exists(self._done_flag_file):
            os.remove(self._done_flag_file)
            
        # 一時ディレクトリを削除
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                os.rmdir(self._temp_dir)
                if self.verbose:
                    print(f"一時ディレクトリを削除しました: {self._temp_dir}")
            except:
                pass
                
        self.shared_memories = {}
        self.semaphores = {}
        
    def __del__(self):
        """デストラクタ"""
        self.cleanup()

class MLaSO_d:
    """ML-assisted Shape Optimization - Deterministic"""
    
    def __init__(self, verbose=False):
        self.params = np.array([
            2.1e11,    # Young's modulus
            0.31,      # Poisson's ratio
            0.5,       # for volume constraint
            0.1,       # for volume constraint
            0.03,      # for volume constraint
            0.001,     # min value of rho
            1.0,       # max value of rho
            0.04,      # regularization parameter
            0.0        # routine index in MLaSO-d
        ], dtype=np.float64)
        
        self.density = None
        self.results = None
        self.verbose = verbose
        self.shm_manager = SharedMemoryLinux(verbose=verbose)
        
        # 共有メモリ名の設定
        self.shm_params_name = "freefem_params"
        self.shm_density_name = "freefem_density"
        self.shm_results_name = "freefem_results"
        
    def initialize_density(self, n_elements):
        """密度場を初期化"""
        self.density = np.ones(n_elements, dtype=np.float64)
        if self.verbose:
            print(f"密度場を初期化しました: {self.density.shape}")
        return self.density
        
    def on_the_job_training(self, iteration=0):
        """FreeFEMを使用して最適化の一反復を実行"""
        if self.density is None:
            print("エラー: 密度場が初期化されていません")
            return False
            
        n_elements = self.density.size
        
        try:
            # フラグファイルのパスを取得
            ready_flag_file, done_flag_file = self.shm_manager.create_flag_files()
            
            # 共有メモリを作成
            params_shm = self.shm_manager.create_shared_memory(
                self.shm_params_name, self.params.nbytes)
            density_shm = self.shm_manager.create_shared_memory(
                self.shm_density_name, self.density.nbytes)
            results_shm = self.shm_manager.create_shared_memory(
                self.shm_results_name, (n_elements + 2) * 8)  # float64は8バイト
                
            if not params_shm or not density_shm or not results_shm:
                print("共有メモリの作成に失敗しました")
                return False
                
            # 共有メモリにデータを書き込む
            self.shm_manager.write_to_shared_memory(self.shm_params_name, self.params)
            self.shm_manager.write_to_shared_memory(self.shm_density_name, self.density)
            
            # 準備完了フラグを作成
            self.shm_manager.create_ready_flag()
            
            # カレントディレクトリを取得
            current_dir = os.getcwd().replace("\\", "/")  # Windowsパスを変換
            
            # FreeFEMコマンドを構築
            cmd = [
                "FreeFem++", "src/2Delastic-FEA-shm.edp",
                "-shm_params", self.shm_params_name,
                "-shm_density", self.shm_density_name,
                "-shm_results", self.shm_results_name,
                "-ne", str(n_elements),
                "-dir", current_dir,
                "-ready_flag", ready_flag_file,
                "-done_flag", done_flag_file
            ]
            
            if self.verbose:
                print(f"FreeFEMコマンドを実行: {' '.join(cmd)}")
                
            # サブプロセスとしてFreeFEMを実行
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
            # 完了フラグを待機
            success = self.shm_manager.wait_for_done_flag()
            
            if not success:
                stdout, stderr = process.communicate()
                print("FreeFEM実行エラー:")
                print(stdout)
                print(stderr)
                return False
                
            # 結果を共有メモリから読み込む
            self.results = self.shm_manager.read_from_shared_memory(
                self.shm_results_name, shape=(n_elements + 2,))
                
            # FreeFEMプロセスの終了を待機
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"FreeFEM実行エラー (コード: {process.returncode}):")
                print(stdout)
                print(stderr)
                return False
                
            if self.verbose:
                print("FreeFEM実行出力:")
                print(stdout)
                
                if stderr:
                    print("FreeFEMエラー出力:")
                    print(stderr)
                    
            # 結果を解析
            if self.results is not None and self.results.size == n_elements + 2:
                gradient = self.results[:n_elements]
                compliance = self.results[n_elements]
                lamG = self.results[n_elements + 1]
                
                if self.verbose:
                    print(f"計算結果:")
                    print(f"  コンプライアンス: {compliance}")
                    print(f"  ラグランジュ乗数: {lamG}")
                    print(f"  勾配ノルム: {np.linalg.norm(gradient)}")
                    
                # パラメータを更新
                self.params[3] = lamG  # ラグランジュ乗数を更新
                
                # 密度場を更新（単純勾配法の例）
                step_size = 0.1
                self.density -= step_size * gradient
                self.density = np.clip(self.density, self.params[5], self.params[6])  # 密度の制約
                
                return True
            else:
                print("結果の読み取りに失敗しました")
                return False
                
        except Exception as e:
            print(f"on_the_job_training実行エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 共有メモリをクリーンアップ
            self.shm_manager.cleanup()
            
    def run_optimization(self, n_elements=5883, n_iterations=10):
        """最適化プロセスを実行"""
        # 密度場を初期化
        self.initialize_density(n_elements)
        
        for i in range(n_iterations):
            print(f"\n反復 {i+1}/{n_iterations}")
            success = self.on_the_job_training(iteration=i)
            
            if not success:
                print(f"反復 {i+1} でエラーが発生しました。最適化を中止します。")
                break
                
            # 中間結果の表示
            if self.verbose:
                density_min = np.min(self.density)
                density_max = np.max(self.density)
                density_avg = np.mean(self.density)
                print(f"密度場の統計: 最小={density_min:.6f}, 最大={density_max:.6f}, 平均={density_avg:.6f}")
                
        print("最適化が完了しました")

def main():
    parser = argparse.ArgumentParser(description='共有メモリを使用したML支援形状最適化')
    parser.add_argument('--verbose', action='store_true', help='詳細出力を有効化')
    parser.add_argument('--iterations', type=int, default=10, help='最適化反復回数')
    parser.add_argument('--elements', type=int, default=5883, help='要素数')
    args = parser.parse_args()
    
    optimizer = MLaSO_d(verbose=args.verbose)
    optimizer.run_optimization(n_elements=args.elements, n_iterations=args.iterations)

if __name__ == "__main__":
    main() 