import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class BDAE(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size):
        super(BDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid() 
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return bottleneck, reconstruction

def add_noise(data, noise_factor=0.1):
    noise = noise_factor * torch.randn_like(data)
    noisy_data = data + noise
    return torch.clamp(noisy_data, 0., 1.) 

def train_bdae(bdae_model, train_data, epochs=50, batch_size=32, noise_factor=0.2, lr=0.001):
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(bdae_model.parameters(), lr=lr)
    bdae_model.train()

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (batch,) in enumerate(train_loader):
            noisy_batch = add_noise(batch, noise_factor)
            
            optimizer.zero_grad()
            _, reconstruction = bdae_model(noisy_batch)
            loss = criterion(reconstruction, batch)
            loss.backward()  
            optimizer.step() 
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def main():
    subject_accuracies = []

    for subject_id in range(1, 13):
        print(f"Processing Subject {subject_id}...")

        eeg_train_data = np.load(f'dataset/{subject_id}/train_data_eeg.npy')
        eye_train_data = np.load(f'dataset/{subject_id}/train_data_eye.npy')
        eeg_test_data = np.load(f'dataset/{subject_id}/test_data_eeg.npy')
        eye_test_data = np.load(f'dataset/{subject_id}/test_data_eye.npy')
        train_labels = np.load(f'dataset/{subject_id}/train_label.npy')
        test_labels = np.load(f'dataset/{subject_id}/test_label.npy')

        scaler_eeg = StandardScaler()
        scaler_eye = StandardScaler()
        eeg_train_data = scaler_eeg.fit_transform(eeg_train_data)
        eye_train_data = scaler_eye.fit_transform(eye_train_data)
        eeg_test_data = scaler_eeg.transform(eeg_test_data)
        eye_test_data = scaler_eye.transform(eye_test_data)

        eeg_train_data = torch.tensor(eeg_train_data, dtype=torch.float32)
        eye_train_data = torch.tensor(eye_train_data, dtype=torch.float32)
        eeg_test_data = torch.tensor(eeg_test_data, dtype=torch.float32)
        eye_test_data = torch.tensor(eye_test_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        eeg_bdae = BDAE(input_size=eeg_train_data.shape[1], hidden_size=256, bottleneck_size=32)
        eye_bdae = BDAE(input_size=eye_train_data.shape[1], hidden_size=256, bottleneck_size=32)

        print("Training EEG BDAE...")
        train_bdae(eeg_bdae, eeg_train_data)
        print("Training Eye BDAE...")
        train_bdae(eye_bdae, eye_train_data)

        eeg_bottleneck, _ = eeg_bdae(eeg_train_data)
        eye_bottleneck, _ = eye_bdae(eye_train_data)
        eeg_test_bottleneck, _ = eeg_bdae(eeg_test_data)
        eye_test_bottleneck, _ = eye_bdae(eye_test_data)

        eeg_bottleneck = eeg_bottleneck.detach()
        eye_bottleneck = eye_bottleneck.detach()
        eeg_test_bottleneck = eeg_test_bottleneck.detach()
        eye_test_bottleneck = eye_test_bottleneck.detach()

        train_features = torch.cat((eeg_bottleneck, eye_bottleneck), dim=1)
        test_features = torch.cat((eeg_test_bottleneck, eye_test_bottleneck), dim=1)

        classifier = nn.Sequential(
            nn.Linear(train_features.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, len(torch.unique(train_labels)))  
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

        classifier.train()
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = classifier(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
        print(f"Subject {subject_id} Classifier Trained.")

        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(test_features)
            _, predictions = torch.max(test_outputs, 1)
            accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
            subject_accuracies.append(accuracy)
            print(f"Subject {subject_id} Accuracy: {accuracy * 100:.2f}%")

    average_accuracy = np.mean(subject_accuracies)
    print(f"Average Accuracy Across All Subjects: {average_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
