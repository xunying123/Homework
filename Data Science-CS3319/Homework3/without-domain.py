import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_data(data_path):
    all_data, all_labels, groups = [], [], []
    for group_id, folder in enumerate(sorted(os.listdir(data_path))):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            data = np.load(os.path.join(folder_path, "data.npy")).reshape(-1, 310)
            labels = np.load(os.path.join(folder_path, "label.npy"))

            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            all_data.append(data)
            all_labels.append(labels)
            groups.extend([group_id] * len(data))
    return torch.tensor(np.vstack(all_data), dtype=torch.float32), torch.tensor(np.hstack(all_labels), dtype=torch.long), np.array(groups)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2, 1))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 15 * 5, 256) 

    def forward(self, x):
        x = x.view(-1, 1, 62, 5)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return x

class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 3) 
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionClassificationModel(nn.Module):
    def __init__(self):
        super(EmotionClassificationModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.emotion_classifier = EmotionClassifier()
        
    def forward(self, x):
        features = self.feature_extractor(x)
        emotion_output = self.emotion_classifier(features)
        return emotion_output

def extract_features(model, data_loader, device):
    model.eval()
    features_list = []
    labels_list = []
    domains_list = []
    
    with torch.no_grad():
        for data, labels, domains in data_loader:
            data = data.to(device)
            features = model.feature_extractor(data)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            domains_list.append(domains.numpy())
    
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    domains = np.hstack(domains_list)
    
    return features, labels, domains

def train_emotion_classifier(model, source_loader, test_loader, epochs=100, device='cuda'):
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU")
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in source_loader:
            data, labels, _ = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        test_acc = evaluate_model(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict()
            trigger_times = 0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    return best_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def collect_and_visualize_features(all_features, all_domains, title='All Folds Feature Visualization'):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)

    unique_domains = np.unique(all_domains)

    cmap = plt.get_cmap('tab10') 
    if len(unique_domains) > 10:
        cmap = plt.get_cmap('tab20')

    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=all_domains, cmap=cmap, alpha=0.7)

    handles, labels = scatter.legend_elements()
    legend_labels = [f"Domain {int(label)}" for label in unique_domains]
    plt.legend(handles[:len(unique_domains)], legend_labels, title="Domains", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('features_tsne222.png')

def run_experiment(data_path='./dataset', epochs=100, visualize_fold=None):
    all_data, all_labels, groups = load_data(data_path)
    
    logo = LeaveOneGroupOut()
    logo_split = logo.split(all_data, all_labels, groups)
    
    accuracies = []
    
    print("运行情感分类实验")
    
    for fold, (train_idx, test_idx) in enumerate(logo_split):
        print(f"\n正在测试被试 {fold + 1} 作为未见域...")

        train_data, train_labels = all_data[train_idx], all_labels[train_idx]
        test_data, test_labels = all_data[test_idx], all_labels[test_idx]

        train_dataset = TensorDataset(train_data, train_labels, torch.tensor(groups[train_idx]))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = TensorDataset(test_data, test_labels, torch.tensor(groups[test_idx]))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = EmotionClassificationModel()

        acc = train_emotion_classifier(model, train_loader, test_loader, epochs=epochs, device='cuda')
        accuracies.append(acc)
        
        print(f"被试 {fold + 1} 的测试准确率: {acc:.2f}%")

        if visualize_fold is not None and fold == visualize_fold:
            print(f"\n正在对折 {fold + 1} 的训练集进行特征可视化...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            features, labels, domains = extract_features(model, train_loader, device)

            collect_and_visualize_features(features, domains, 
                                           title=f'Fold {fold + 1} Training Set Feature Visualization with t-SNE')

    print("\n实验结束。最终结果：")
    print("\n情感分类准确率：")
    for i, acc in enumerate(accuracies):
        print(f"被试 {i+1}: {acc:.2f}%")
    print(f"平均准确率: {np.mean(accuracies):.2f}%")
    print(f"准确率标准差: {np.std(accuracies):.2f}%")

if __name__ == "__main__":
    run_experiment(data_path='./dataset', epochs=100, visualize_fold=2)
