import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.autograd import Function

# Gradient Reversal Layer
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambda_, None

# Feature Extractor: Simple MLP
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_dim=512):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

# Classifier: Emotion Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Domain Discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Custom Dataset for SEED data
class SEEDDataset(Dataset):
    def __init__(self, data_path, subject_idx, transform=None):
        self.data_path = data_path
        self.subject_idx = subject_idx
        self.transform = transform
        self.data = None
        self.labels = None
        self.load_data()

    def load_data(self):
        subject_data = np.load(os.path.join(self.data_path, f'{self.subject_idx+1}/data.npy'))
        subject_labels = np.load(os.path.join(self.data_path, f'{self.subject_idx+1}/label.npy'))
        self.data = subject_data.reshape(-1, 310)  # Reshape to (n_samples, 310)
        self.labels = subject_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# DANN Model (Feature Extractor + Classifier + Domain Discriminator)
class DANN(nn.Module):
    def __init__(self, input_dim=310, num_classes=3):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        self.classifier = Classifier(input_dim=128, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim=128)

    def forward(self, x, lambda_=1.0):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        domain_output = self.domain_discriminator(features)
        return class_output, domain_output

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

# Training function for one epoch
def train_epoch(model, data_loader, optimizer, criterion_class, criterion_domain, lambda_):
    model.train()
    total_loss_class = 0
    total_loss_domain = 0
    correct_class = 0
    total_class = 0
    for data, label in data_loader:
        data = data.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        # Forward pass
        class_output, domain_output = model(data, lambda_)

        # Classification loss
        loss_class = criterion_class(class_output, label)
        
        # Domain loss (domain adaptation)
        domain_label = torch.zeros(data.size(0)).cuda()  # Use 0 for source domain
        loss_domain = criterion_domain(domain_output.view(-1), domain_label)

        # Total loss
        loss = loss_class + loss_domain
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss_class += loss_class.item()
        total_loss_domain += loss_domain.item()

        _, predicted = torch.max(class_output.data, 1)
        correct_class += (predicted == label).sum().item()
        total_class += label.size(0)

    accuracy = 100 * correct_class / total_class
    return total_loss_class / len(data_loader), total_loss_domain / len(data_loader), accuracy

# Cross-validation (Leave-One-Subject-Out)
def cross_validation(data_path, num_subjects=12, num_epochs=20, lambda_=1.0):
    accuracies = []

    for subject in range(num_subjects):
        print(f"Training for subject {subject+1}...")

        # Prepare training and testing datasets
        train_subjects = [i for i in range(num_subjects) if i != subject]
        test_subject = subject

        # Load training and testing data
        train_data = []
        train_labels = []
        for idx in train_subjects:
            dataset = SEEDDataset(data_path, idx)
            train_data.append(dataset)
        test_data = SEEDDataset(data_path, test_subject)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Initialize model
        model = DANN(input_dim=310, num_classes=3).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.BCELoss()

        # Training loop
        for epoch in range(num_epochs):
            lambda_ = min(1.0, (epoch + 1) / num_epochs)
            train_loss_class, train_loss_domain, train_accuracy = train_epoch(
                model, train_loader, optimizer, criterion_class, criterion_domain, lambda_
            )
            print(f"Epoch {epoch+1}/{num_epochs}, Class Loss: {train_loss_class:.4f}, Domain Loss: {train_loss_domain:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Testing
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in test_loader:
                data = data.cuda()
                label = label.cuda()

                class_output, _ = model(data)
                _, predicted = torch.max(class_output.data, 1)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Subject {subject+1} Test Accuracy: {accuracy*100:.2f}%")
        accuracies.append(accuracy)

    print(f"Average Test Accuracy: {np.mean(accuracies)*100:.2f}%")
    return accuracies

# Main function
if __name__ == '__main__':
    data_path = './dataset'  # Change this to your dataset path
    accuracies = cross_validation(data_path)
    print(f"Final Accuracy across all subjects: {np.mean(accuracies)*100:.2f}%")
