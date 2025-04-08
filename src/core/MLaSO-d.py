import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
import numpy as np
import yaml
import time
import tempfile
import os
import subprocess
import argparse

# FreeFEMのパスを明示的に設定
import pyfreefem as pf
pf.FreeFemPath = "C:\\Program Files (x86)\\FreeFem++\\FreeFem++.exe"  # 4.11のパスを指定

from sensitivity_net import (
    sensitivity_prediction,
    sensitivity_prediction_2,
    sensitivity_prediction_fc,
)

class MLaSO_d:
    def __init__(self, verbose=False):
        """
        MLaSO-dのパラメータを読み込む
        
        Args:
            verbose (bool): 詳細な出力を表示するかどうか
        """
        self.verbose = verbose
        
        with open("./config/MLaSO-d_param.yaml", "r") as yml:
            config = yaml.safe_load(yml)

        self.beta_min = config["MLaSO_d"]["beta_min"]
        self.k = config["MLaSO_d"]["k"]
        self.ks = config["MLaSO_d"]["ks"]
        self.kr = config["MLaSO_d"]["kr"]
        self.kp = config["MLaSO_d"]["kp"]
        self.max_loop = config["MLaSO_d"]["max_loop"]
        self.Ne = config["MLaSO_d"]["Ne"]
        self.r_min = config["MLaSO_d"]["r_min"]

        self.max_epoch = config["on_the_job_training"]["max_epoch"]
        self.constant_epoch = config["on_the_job_training"]["constant_epoch"]
        self.gamma = config["on_the_job_training"]["gamma_0"]
        self.delta_gamma = config["on_the_job_training"]["delta_gamma"]
        self.R_min = config["on_the_job_training"]["R_min"]
        self.E_min = config["on_the_job_training"]["E_min"]
        self.warmup_epoch = config["on_the_job_training"]["warmup_epoch"]
        self.lr = config["on_the_job_training"]["lr"]
        self.sub_lr = config["on_the_job_training"]["sub_lr"]
        self.constant_lr = config["on_the_job_training"]["constant_lr"]
        self.tau_t = config["on_the_job_training"]["tau_t"]

        self.rho = torch.ones([self.Ne], dtype=torch.float32)
        self.d = torch.zeros([self.Ne], dtype=torch.float32)
        self.s = torch.zeros([2, 2], dtype=torch.float32)
        self.er = torch.zeros([self.Ne], dtype=torch.float32)
        self.epsilon = torch.zeros([self.Ne], dtype=torch.float32)
        self.d_bar_k = torch.ones([self.Ne], dtype=torch.float32)
        self.d_bar_kr = torch.ones([self.Ne], dtype=torch.float32)

        self.lamG = 0.1

        self.comp = []

        # FreeFEM++の実行ファイルのパスを設定
        self.freefem_path = "C:\\Program Files (x86)\\FreeFem++\\FreeFem++.exe"
        
        # メモリマップトファイルのパス
        self.temp_dir = tempfile.mkdtemp(prefix="mlaso_")
        self.input_file = os.path.join(self.temp_dir, "input.txt")
        self.output_file = os.path.join(self.temp_dir, "output.txt")
        self.params_file = os.path.join(self.temp_dir, "params.txt")
        
        # メモリマップトファイルの初期化
        self.setup_mmap_files()
        
        if self.verbose:
            print(f"初期化完了：要素数 = {self.Ne}, 最大イテレーション = {self.max_loop}")
            print(f"一時ディレクトリ: {self.temp_dir}")

    def setup_mmap_files(self):
        """メモリマップトファイルの初期化"""
        # 入力用のファイル（密度場用）
        with open(self.input_file, "w") as f:
            f.write("0\n" * self.Ne)
        
        # 出力用のファイル（勾配とコンプライアンス用）
        with open(self.output_file, "w") as f:
            f.write("0\n" * (self.Ne + 2))
        
        # パラメータ用のファイル
        with open(self.params_file, "w") as f:
            f.write("0\n" * 10)

    def cleanup_mmap_files(self):
        """メモリマップトファイルの削除"""
        try:
            os.remove(self.input_file)
            os.remove(self.output_file)
            os.remove(self.params_file)
            os.rmdir(self.temp_dir)
            if self.verbose:
                print(f"一時ファイルを削除しました: {self.temp_dir}")
        except Exception as e:
            print(f"一時ファイルの削除中にエラーが発生: {e}")

    def on_the_job_training(self, model) -> None:
        """
        FreeFemを用いてFEMを実行し、その結果を利用して機械学習モデルを訓練する
        """
        # 密度場をファイルに書き込み
        rho_np = self.rho.numpy().astype(np.float64)
        if self.verbose:
            print(f"\n密度場の書き込み:")
            print(f"ファイル: {self.input_file}")
            print(f"要素数: {len(rho_np)}")
            print(f"最小値: {np.min(rho_np)}")
            print(f"最大値: {np.max(rho_np)}")
            print(f"平均値: {np.mean(rho_np)}")
            
        with open(self.input_file, "w") as f:
            for val in rho_np:
                f.write(f"{val}\n")  # 通常の浮動小数点数表記で書き込み
        
        # パラメータをファイルに書き込み
        params = np.array([
            210e9,  # E
            0.31,   # nu
            0.5,    # Volcoef
            self.lamG,  # lamG
            0.03,   # sigG
            0.001,  # rhomin
            1.0,    # rhomax
            0.04,   # regR
            float(self.kr)  # kr
        ], dtype=np.float64)
        
        if self.verbose:
            print(f"\nパラメータの書き込み:")
            print(f"ファイル: {self.params_file}")
            for i, val in enumerate(params):
                print(f"params[{i}] = {val:e}")
                
        with open(self.params_file, "w") as f:
            for val in params:
                f.write(f"{val:e}\n")  # 指数表記で書き込み
        
        # FreeFEMを実行
        try:
            if self.verbose:
                print("\nFreeFEMの実行:")
                print(f"実行ファイル: FreeFem++")
                print(f"スクリプト: src/2Delastic-FEA.edp")
                print("引数:")
                print(f"  -in {os.path.abspath(self.input_file)}")
                print(f"  -out {os.path.abspath(self.output_file)}")
                print(f"  -param {os.path.abspath(self.params_file)}")
                print(f"  -ne {self.Ne}")
                print(f"  -dir {os.getcwd()}")
            
            result = subprocess.run(
                [
                    "C:\\Program Files (x86)\\FreeFem++\\FreeFem++.exe",  # 絶対パスを指定
                    "src/2Delastic-FEA.edp",
                    "-in", os.path.abspath(self.input_file),
                    "-out", os.path.abspath(self.output_file),
                    "-param", os.path.abspath(self.params_file),
                    "-ne", str(self.Ne),
                    "-dir", os.getcwd()
                ],
                check=True,
                capture_output=True,
                text=True
            )
            
            if self.verbose:
                print("\nFreeFEMの出力:")
                print(result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"FreeFEMの実行中にエラーが発生: {e}")
            print(f"標準出力: {e.stdout}")
            print(f"標準エラー: {e.stderr}")
            raise
        
        # 結果をファイルから読み取り
        output = []
        with open(self.output_file, "r") as f:
            for line in f:
                output.append(float(line.strip()))
        output = np.array(output)
        
        # 感度勾配と制約値を設定
        self.d = torch.tensor(output[:self.Ne], dtype=torch.float32)
        self.comp.append(float(output[self.Ne]))  # コンプライアンス値
        self.lamG = float(output[self.Ne + 1])    # ラグランジュ乗数
        
        # 4 ---------------------------------------------------------------------
        self.s[1] = self.s[0].clone()
        self.s[0] = torch.tensor(
            [torch.max(self.d), torch.min(self.d)]
        )  # index : max->0 , min->1

        # temporal variables
        d_tilde_k = (self.d - self.s[0, 1]) / (self.s[0, 0] - self.s[0, 1])
        d_tilde_kr = (self.d - self.s[0, 1]) / (self.s[0, 0] - self.s[0, 1])
        
        if self.verbose:
            print(
                "d->min:",
                self.d.min(),
                "max:",
                self.d.max(),
                "mean:",
                torch.mean(self.d),
                "var:",
                torch.var(self.d),
            )
            print("s->min:", self.s[0, 1], "max:", self.s[0, 0])
            print(
                "d_tiled_k->min:",
                d_tilde_kr.min(),
                "max:",
                d_tilde_kr.max(),
                "mean:",
                torch.mean(d_tilde_kr),
                "var:",
                torch.var(d_tilde_kr),
            )

        # 6 ---------------------------------------------------------------------
        if self.k > 0:
            # 7 ---------------------------------------------------------------------
            delta = self.d_bar_k.clone().detach().to(torch.float32)

            # 9 ---------------------------------------------------------------------
            if self.kr >= 2:
                # 10 ---------------------------------------------------------------------
                self.er = torch.max(
                    torch.ones([self.Ne]),
                    self.er.clone()
                    + self.gamma
                    * (self.epsilon - torch.min(self.epsilon))
                    / (torch.max(self.epsilon) - torch.min(self.epsilon)),
                )

            # 13 ---------------------------------------------------------------------
            beta = torch.max(
                torch.ones([self.Ne]) * self.beta_min,
                torch.min(torch.ones([self.Ne]), 1 / self.er),
            )
            beta = torch.clamp(beta, min=1e-6, max=1)
            
            if self.verbose:
                print(
                    "beta->min:", beta.min(), "max:", beta.max(), "mean:", torch.mean(beta)
                )

            # 14 ---------------------------------------------------------------------
            d_bar_k = d_tilde_kr * beta + self.d_bar_kr * (1 - beta)
            self.d_bar_k, self.d_bar_kr = (
                d_bar_k.clone().detach().to(torch.float32),
                d_bar_k.clone().detach().to(torch.float32),
            )

            # 15 ---------------------------------------------------------------------
            pred_error = model(delta) - self.d_bar_k
            self.epsilon = torch.abs(pred_error)
            
            if self.verbose:
                print(
                    "d_bar_k-> min : ",
                    self.d_bar_k.min(),
                    "max : ",
                    self.d_bar_k.max(),
                    "mean : ",
                    torch.mean(self.d_bar_k),
                    "var : ",
                    torch.var(self.d_bar_k),
                )
        # 16 ---------------------------------------------------------------------
        else:
            # 17 ---------------------------------------------------------------------
            self.d_bar_k, self.d_bar_kr = (
                d_tilde_k.clone().detach().to(torch.float32),
                d_tilde_kr.clone().detach().to(torch.float32),
            )

        # 19 ---------------------------------------------------------------------
        model.train()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=0,  # パラメータは適当
        )
        # optimizer = RAdamScheduleFree(model.parameters(),silent_sgd_phase=True)
        # optimizer.train()

        # 最初の訓練時にbaseとminの値を決める
        if self.k - self.ks == 0:
            base_lr = self.lr
            min_lr = self.constant_lr

        # 論文には記載されていないが、感度の変化が大きい最初の数ステップは切り捨てる
        if self.k >= self.ks:
            for i in range(self.max_epoch):
                if self.k - self.ks == 0:
                    if i < self.warmup_epoch:
                        self.lr = (base_lr - min_lr) * i / self.warmup_epoch + min_lr
                    elif (
                        i >= self.warmup_epoch
                        and i < self.max_epoch - self.constant_epoch
                    ):
                        self.lr = (base_lr - min_lr) * np.cos(
                            0.5 * np.pi * i / (self.max_epoch - self.constant_epoch)
                        ) + min_lr
                    else:
                        self.lr = min_lr
                else:
                    self.lr = np.cos(0.5 * np.pi * i / (self.max_loop)) * self.sub_lr
                # 20 ---------------------------------------------------------------------
                optimizer.zero_grad()
                pred = model(delta)
                loss = loss_fn(pred, self.d_bar_k)

                # エポックの途中経過は詳細モードでのみ表示、または10エポックごとに表示
                if self.verbose or i % 10 == 0:
                    print(
                        f"  エポック {i}: Loss = {loss.item():.6e}, LR = {self.lr:.6e}"
                    )
                    
                loss.backward()
                optimizer.step()

                # 21,22 ---------------------------------------------------------------------
                if loss <= self.tau_t:
                    print(f"  収束条件到達 (損失 {loss.item():.6e} <= {self.tau_t})")
                    break

        if self.k - self.ks == 0:
            self.max_epoch = 100

        self.d = self.s[0, 1] + (self.d_bar_k - torch.min(self.d_bar_k)) / (
            torch.max(self.d_bar_k) - torch.min(self.d_bar_k)
        ) * (self.s[0, 0] - self.s[0, 1])
        
        self.kr += 1

    def on_the_job_prediction(self, model):
        """
        学習させたモデルを利用して、On-the-job predictionを実行
        """
        ws = (self.s[0, 0] - self.s[0, 1]) / (self.s[1, 0] - self.s[1, 1])
        bs = (self.s[0, 1] + self.s[0, 0] - ws * (self.s[1, 1] + self.s[1, 0])) / 2
        s_hat = ws * self.s[0] + bs

        model.eval()
        test = self.d_bar_k.clone().detach().to(torch.float32)
        with torch.no_grad():
            self.d_bar_k = model(test)
        self.d = s_hat[1] + (self.d_bar_k - torch.min(self.d_bar_k)) / (
            torch.max(self.d_bar_k) - torch.min(self.d_bar_k)
        ) * (s_hat[0] - s_hat[1])

        if self.verbose:
            print(
                f"{self.kp}回目の予測->感度:",
                self.d,
                "min:",
                self.d.min(),
                "max:",
                self.d.max(),
                "mean:",
                torch.mean(self.d),
            )
        else:
            print(
                f"  感度予測: min={self.d.min():.4e}, max={self.d.max():.4e}, mean={torch.mean(self.d):.4e}"
            )
            
        self.kp += 1

    def implementation(self):
        """
        MLaSO-dの実行
        """
        print("===== MLaSO-d トポロジー最適化開始 =====")
        print(f"要素数: {self.Ne}, 最大イテレーション: {self.max_loop}")
        
        # model = sensitivity_prediction(self.Ne, self.R_min, self.E_min)
        model = sensitivity_prediction_2(self.Ne, self.R_min)
        # model = sensitivity_prediction_fc(self.Ne)

        dt = 0.2
        diff = 0.01

        # 結果を保存するための配列
        compliance_history = []
        volume_history = []
        
        for self.k in range(self.max_loop):
            print(f"\n[イテレーション {self.k}/{self.max_loop-1}]")
            
            if self.k <= self.ks or np.mod(np.max([self.k - self.ks, 1]), 2) == 0:
                print(f"実行: On-the-job training (回数: {self.kr})")
                self.on_the_job_training(model)
            else:
                print(f"実行: On-the-job prediction (回数: {self.kp})")
                self.on_the_job_prediction(model)

            # 必要な指標を出力
            compliance = self.comp[-1]
            compliance_history.append(compliance)
            
            volume_ratio = torch.mean(self.rho).item()
            volume_history.append(volume_ratio)
            
            print(f"コンプライアンス: {compliance:.6e}")
            print(f"体積比: {volume_ratio:.4f}")

            # 密度場の更新
            self.rho = torch.clamp(self.rho + dt * self.d, min=0.001, max=1.0)
            
            if self.verbose:
                print(
                    "rho->",
                    self.rho,
                    "min:",
                    self.rho.min(),
                    "max:",
                    self.rho.max(),
                    "mean:",
                    torch.mean(self.rho),
                )
            else:
                print(
                    f"密度場: min={self.rho.min():.4f}, max={self.rho.max():.4f}, mean={torch.mean(self.rho):.4f}"
                )

            # 結果をファイルに保存
            os.makedirs("./data/sensitivity", exist_ok=True)
            os.makedirs("./data/rho", exist_ok=True)
            
            np.savetxt(
                f"./data/sensitivity/sensitivity_{self.k}.csv",
                self.d.numpy(),
                delimiter=",",
            )
            np.savetxt(f"./data/rho/rho_{self.k}.csv", self.rho.numpy(), delimiter=",")

            # update dynamic weightage
            if self.kr >= 4:
                if (
                    self.comp[self.kr - 1] != self.comp[self.kr - 2]
                    and self.comp[self.kr - 2] != self.comp[self.kr - 3]
                ):
                    diff_obj_kr = abs(
                        (self.comp[self.kr - 1] - self.comp[self.kr - 2])
                        / abs(self.comp[self.kr - 1] - self.comp[self.kr - 2])
                        + (self.comp[self.kr - 2] - self.comp[self.kr - 3])
                        / abs(self.comp[self.kr - 2] - self.comp[self.kr - 3])
                    )
                    diff_obj_kr_1 = abs(
                        (self.comp[self.kr - 1] - self.comp[self.kr - 2])
                        / abs(self.comp[self.kr - 1] - self.comp[self.kr - 2])
                        - (self.comp[self.kr - 2] - self.comp[self.kr - 3])
                        / abs(self.comp[self.kr - 2] - self.comp[self.kr - 3])
                    )
                    if diff_obj_kr <= diff and diff_obj_kr_1 <= diff:
                        self.gamma = self.gamma + self.delta_gamma
                        if self.verbose:
                            print(f"ガンマ値を更新: {self.gamma}")
                            
            print("-------------------------------------------------------")
        
        print("\n===== 最適化完了 =====")
        print(f"最終コンプライアンス: {compliance_history[-1]:.6e}")
        print(f"最終体積比: {volume_history[-1]:.4f}")
        
        # 履歴データを保存
        os.makedirs("./src/SaveData", exist_ok=True)
        history_data = np.column_stack((
            np.arange(len(compliance_history)), 
            compliance_history, 
            volume_history
        ))
        np.savetxt("./src/SaveData/history.dat", history_data, 
                  header="Iteration Compliance Volume", comments='')
        
        print(f"履歴データを保存しました: ./src/SaveData/history.dat")

    def __del__(self):
        """
        クラスのインスタンスが削除される際に呼び出される
        """
        self.cleanup_mmap_files()

def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description='MLaSO-d トポロジー最適化')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='詳細な出力を表示する')
    return parser.parse_args()

if __name__ == "__main__":
    # コマンドライン引数を解析
    args = parse_args()
    
    start = time.time()
    mlaso = MLaSO_d(verbose=args.verbose)
    mlaso.implementation()
    end = time.time()
    
    duration = end - start
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"計算時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")
