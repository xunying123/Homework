import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d((2, 1))
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 15 * 5, 256) 

    def forward(self, x):
        x = x.view(-1, 1, 62, 5)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return x

class DomainClassifier(nn.Module):
    def __init__(self, num_domains):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_domains)
        
    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

class DomainAdaptation(nn.Module):
    def __init__(self, num_domains):
        super(DomainAdaptation, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.emotion_classifier = EmotionClassifier()
        self.domain_classifier = DomainClassifier(num_domains)
        
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        emotion_output = self.emotion_classifier(features)
        domain_output = self.domain_classifier(features, alpha)
        return emotion_output, domain_output, features

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

def train_domain_adaptation(model, source_loader, target_loader, test_loader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    emotion_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for (source_data, source_labels, source_domains), (target_data, _, target_domains) in zip(source_loader, target_loader):
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            source_domains = source_domains.to(device)
            
            target_data = target_data.to(device)
            target_domains = target_domains.to(device)
            
            optimizer.zero_grad()
            
            p = float(epoch) / epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            emotion_output, domain_output, _ = model(source_data, alpha)
            emotion_loss = emotion_criterion(emotion_output, source_labels)
            domain_loss_source = domain_criterion(domain_output, source_domains)
            
            _, domain_output_target, _ = model(target_data, alpha)
            domain_loss_target = domain_criterion(domain_output_target, target_domains)
            
            loss = emotion_loss + domain_loss_source + domain_loss_target
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        test_acc = evaluate_model(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            
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
            
            emotion_output, _, _ = model(data)
            _, predicted = torch.max(emotion_output.data, 1)
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
    plt.savefig('features_tsne.png')

def run_experiment_on_gpus(data_path='./dataset', epochs=100, visualize_fold=None):
    all_data, all_labels, groups = load_data(data_path)
    
    logo = LeaveOneGroupOut()
    logo_split = logo.split(all_data, all_labels, groups)
    
    da_accuracies = []
    print("运行域适应实验 (Domain Adaptation) 使用 Leave-One-Group-Out...")
    
    for fold, (train_idx, test_idx) in enumerate(logo_split):
        print(f"\n正在测试被试 {fold + 1} 作为目标域...")
        
        train_data, train_labels = all_data[train_idx], all_labels[train_idx]
        test_data, test_labels = all_data[test_idx], all_labels[test_idx]
        train_domains = groups[train_idx]
        test_domains = groups[test_idx]
        
        source_dataset = TensorDataset(train_data, train_labels, torch.tensor(train_domains))
        source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
        
        target_dataset = TensorDataset(test_data, test_labels, torch.tensor(test_domains))
        target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
        
        test_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)

        num_domains = len(np.unique(groups))
        model = DomainAdaptation(num_domains=num_domains)

        da_acc = train_domain_adaptation(model, source_loader, target_loader, test_loader, epochs=epochs)
        da_accuracies.append(da_acc)
        
        print(f"被试 {fold + 1} 的域适应测试准确率: {da_acc:.2f}%")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if visualize_fold is not None and fold == visualize_fold:
            print(f"\n正在对折 {fold + 1} 的训练集进行特征可视化...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            features, labels, domains = extract_features(model, source_loader, device)

            collect_and_visualize_features(features, domains, 
                                           title=f'Fold {fold + 1} Training Set Feature Visualization with t-SNE')
        
    print("\n实验结束。最终结果：")
    print("\n域适应准确率：")
    for i, acc in enumerate(da_accuracies):
        print(f"被试 {i+1}: {acc:.2f}%")
    print(f"平均准确率: {np.mean(da_accuracies):.2f}%")
    print(f"准确率标准差: {np.std(da_accuracies):.2f}%")
    
if __name__ == "__main__":
    run_experiment_on_gpus(data_path='./dataset', epochs=150, visualize_fold=1)
