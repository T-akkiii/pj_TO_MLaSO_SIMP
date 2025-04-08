import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree  # ScheduleFreeからRAdamScheduleFreeをインポート

# モデルの定義（例として単純な全結合ネットワーク）
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# データセットの準備（例としてランダムなデータを使用）
train_data = torch.randn(1000, 784)  # 1000サンプル、各サンプル784次元
train_labels = torch.randint(0, 10, (1000,))  # 0から9の整数ラベル

# データローダーの作成
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# モデルと損失関数の定義
model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# RAdamScheduleFreeオプティマイザの定義
optimizer = RAdamScheduleFree(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(1000):  # 10エポックの学習
    model.train()
    optimizer.train()  # オプティマイザを訓練モードに設定

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 検証やテスト時にはオプティマイザを評価モードに設定
    model.eval()
    optimizer.eval()
    # ここで検証データを用いた評価を行う

# モデルの保存
torch.save(model.state_dict(), 'simple_model.pth') 