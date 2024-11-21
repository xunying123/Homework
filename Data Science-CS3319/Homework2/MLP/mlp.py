import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# 数据加载
def load_data(data_path):
    all_data, all_labels, groups = [], [], []
    for group_id, folder in enumerate(sorted(os.listdir(data_path))):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            data = np.load(os.path.join(folder_path, "data.npy")).reshape(-1, 310)  # 展平 (62, 5) -> 310
            labels = np.load(os.path.join(folder_path, "label.npy"))
            all_data.append(data)
            all_labels.append(labels)
            groups.extend([group_id] * len(data))
    return torch.tensor(np.vstack(all_data), dtype=torch.float32), \
           torch.tensor(np.hstack(all_labels), dtype=torch.long), \
           torch.tensor(groups, dtype=torch.long)

# 定义模型
class MLP(nn.Module):
    def __init__(self, input_size=310, num_classes=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_model(model, train_data, train_labels, criterion, optimizer, epochs):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

# 测试函数
def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        predictions = torch.argmax(outputs, dim=1)
    return accuracy_score(test_labels.cpu(), predictions.cpu())

# 主流程
def main(data_path):
    X, y, groups = load_data(data_path)
    logo = LeaveOneGroupOut()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx].to(device), X[test_idx].to(device)
        y_train, y_test = y[train_idx].to(device), y[test_idx].to(device)

        # 初始化模型和优化器
        model = MLP(input_size=310, num_classes=len(torch.unique(y))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练和测试
        train_model(model, X_train, y_train, criterion, optimizer, epochs=250)
        acc = evaluate_model(model, X_test, y_test)
        results.append(acc)
        print(f"Fold Accuracy: {acc:.2f}")

    print(f"Mean Accuracy: {np.mean(results):.2f}")
    print(f"Std Accuracy {np.std(results):.2f}")

# 运行
main("dataset")
