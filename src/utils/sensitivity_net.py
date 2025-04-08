import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class sensitivity_prediction(nn.Module):
    def __init__(self, Ne, Rmin, E_min):
        super(sensitivity_prediction, self).__init__()

        # 式に基づく全結合層（入力層から隠れ層）の重み行列を定義（バイアスなし）
        self.fc1_weight = nn.Parameter(torch.empty((Ne, Ne), dtype=torch.float32))

        # 局所的な接続ルール
        self.connection_rule = torch.tensor(
            np.load("./src/mesh/Rij.npy") - Rmin, dtype=torch.float32
        )  # 接続ルール
        self.sparse_indices = (
            torch.argwhere(self.connection_rule < 0).clone().detach()
        )  # 負の値の位置だけ保持
        self.sparse_weights = nn.Parameter(
            torch.empty(self.sparse_indices.size(0), dtype=torch.float32)
        )  # 接続される重みだけを定義

        # He初期化を適用
        self._initialize_sparse_weights()

    def _initialize_sparse_weights(self):
        """He初期化を適用する"""
        # スパース接続用の重み (sparse_weights) を He 初期化に基づいて設定
        fan_in = self.fc1_weight.size(0)  # 入力ノード数
        bound = np.sqrt(2.0 / fan_in)
        with torch.no_grad():
            self.sparse_weights.uniform_(-bound, bound)

    def initialize_fc1_weights(self):
        """fc1の重み行列をカスタム式で初期化"""
        self.wij = np.load("./src/mesh/wij_param.npy")
        with torch.no_grad():
            self.fc1_weight.data = torch.tensor(self.wij, dtype=torch.float32)

    def forward(self, x):
        # x の形状を確認
        if x.ndim == 1:
            x = x.unsqueeze(0)  # バッチ次元を追加

        # 入力層から隠れ層への全結合演算
        try:
            x = x @ self.fc1_weight.T
        except Exception as e:
            print("Error during matrix multiplication")
            raise e

        # スパース接続による隠れ層から出力層への演算
        i_indices = self.sparse_indices[:, 0]  # 隠れ層のインデックス
        j_indices = self.sparse_indices[:, 1]  # 出力層のインデックス

        # スパース行列を構築 (Ne, Ne)
        Ne = self.fc1_weight.size(0)
        indices = torch.stack([i_indices, j_indices])  # shape: [2, #connections]
        values = self.sparse_weights  # shape: [#connections]
        sparse_mat = torch.sparse_coo_tensor(indices, values, (Ne, Ne), device=x.device)
        dense_mat = sparse_mat.to_dense()
        # スパース行列での計算 (batch_size, Ne) @ (Ne, Ne) -> (batch_size, Ne)
        output = x @ dense_mat

        # LeakyReLUを適用
        output = torch.nn.functional.leaky_relu(output, negative_slope=0.001)

        return output


# 上記のモデルよりも高速なモデル
class sensitivity_prediction_2(nn.Module):
    def __init__(self, Ne, Rmin):
        super(sensitivity_prediction_2, self).__init__()

        self.Ne = Ne

        # fc1_weight は通常の Parameter
        self.fc1_weight = nn.Parameter(torch.empty((Ne, Ne), dtype=torch.float32))

        # connection_rule や sparse_indices の生成
        self.connection_rule = torch.tensor(
            np.load("./src/mesh/Rij.npy") - Rmin, dtype=torch.float32
        )
        self.sparse_indices = torch.argwhere(self.connection_rule < 0).clone().detach()

        # スパース接続用重み
        self.sparse_weights = nn.Parameter(
            torch.empty(self.sparse_indices.size(0), dtype=torch.float32)
        )

        # He 初期化
        self._initialize_sparse_weights()

        # スパース行列そのものを属性として一度だけ作る（Parameterではなくbufferがよい場合もある）
        self.register_buffer(
            "sparse_indices_tensor",
            self.sparse_indices.t().to(torch.long),  # shape [2, n_connections]
        )

        # まだ values (sparse_weights) は更新されるものなので、行列には加えない。
        # forward 内で weights だけは毎回差し替えてもよいが、indices は固定なのでOK。

    def _initialize_sparse_weights(self):
        fan_in = self.fc1_weight.size(0)  # 入力ノード数
        bound = np.sqrt(2.0 / fan_in)
        with torch.no_grad():
            self.sparse_weights.uniform_(-bound, bound)

    def initialize_fc1_weights(self):
        self.wij = np.load("./src/mesh/wij_param.npy")
        with torch.no_grad():
            self.fc1_weight.copy_(torch.tensor(self.wij, dtype=torch.float32))

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # 全結合 (Dense)
        x = x @ self.fc1_weight.T

        # -- スパース行列の作成をここで行う --
        # ただし indices は固定で buffer として持っているため再利用し、
        # weights だけを差し込む
        sparse_mat = torch.sparse_coo_tensor(
            self.sparse_indices_tensor,
            self.sparse_weights,
            (self.Ne, self.Ne),
            device=x.device,
        )

        # Sparse x Dense の掛け算を行う
        output = torch.sparse.mm(x, sparse_mat)

        # LeakyReLU
        output = torch.nn.functional.leaky_relu(output, negative_slope=0.001)

        return output


class sensitivity_prediction_fc(nn.Module):
    def __init__(self, Ne):
        super(sensitivity_prediction_fc, self).__init__()
        self.fc1 = nn.Linear(Ne, Ne)  # 入力→隠れ層
        self.fc2 = nn.Linear(Ne, Ne)  # 隠れ層→出力層
        self.relu = nn.ReLU()  # 活性化関数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # 隠れ層の活性化関数
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    matrix = np.load("./src/mesh/Rij.npy")
    # 行列を1次元に変換して要素を取得
    n = 5
    nth_smallest = []
    for i in range(5883):
        nth_smallest.append(np.partition(matrix[i], n - 1)[n - 1])

    # ヒストグラムをプロット
    plt.hist(nth_smallest, bins=100, edgecolor="black")
    plt.title("Histogram of Matrix Elements")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
