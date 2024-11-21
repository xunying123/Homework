import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size=310, hidden_size=128, num_layers=2, num_classes=3):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,  # 每个时间步的特征数
            hidden_size=hidden_size,  # 隐藏层大小
            num_layers=num_layers,  # LSTM层数
            batch_first=True  # 输入形状为 (batch, seq, feature)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        _, (hn, _) = self.rnn(x)  # 仅取最后一层的隐藏状态 hn
        out = self.fc(hn[-1])  # hn[-1] 的形状为 (batch, hidden_size)
        return out

# 数据加载和时间窗口预处理
def load_data_with_windows(data_path, window_size=3):
    all_data, all_labels, groups = [], [], []
    for group_id, folder in enumerate(sorted(os.listdir(data_path))):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            data = np.load(os.path.join(folder_path, "data.npy"))  # 形状 (样本数, 62, 5)
            labels = np.load(os.path.join(folder_path, "label.npy"))

            # 展平通道维度: (样本数, 310)
            data = data.reshape(data.shape[0], -1)

            # 创建时间窗口
            num_windows = data.shape[0] - window_size + 1
            windows = np.array([data[i:i + window_size] for i in range(num_windows)])
            window_labels = labels[window_size - 1:]  # 对应窗口的最后一个样本的标签

            all_data.append(windows)
            all_labels.append(window_labels)
            groups.extend([group_id] * len(window_labels))
    return torch.tensor(np.vstack(all_data), dtype=torch.float32), \
           torch.tensor(np.hstack(all_labels), dtype=torch.long), \
           torch.tensor(groups, dtype=torch.long)

# 创建DataLoader
def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, criterion, optimizer,  device,  epochs = 50,):
    model.train()
    for _ in range(epochs):
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

# 测试函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    return accuracy_score(torch.cat(all_labels), torch.cat(all_predictions))

# 主流程
def main_rnn(data_path):
    X, y, groups = load_data_with_windows(data_path)
    logo = LeaveOneGroupOut()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 创建DataLoader
        train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

        # 初始化RNN模型和优化器
        model = RNN(input_size=X_train.size(2), hidden_size=256, num_layers=3, num_classes=len(torch.unique(y))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练和测试
        train_model(model, train_loader, criterion, optimizer, device)
        acc = evaluate_model(model, test_loader, device)
        results.append(acc)
        print(f"Fold Accuracy: {acc:.2f}")

    print(f"Mean Accuracy: {np.mean(results):.2f}")
    print(f"Std Accuracy: {np.std(results):.2f}")

# 运行
if __name__ == "__main__":
    main_rnn("dataset")
