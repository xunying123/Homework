import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class MLP_EEG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_EEG, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class MLP_Eye(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Eye, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def train(model, optimizer, train_data, train_labels, epochs=100, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

def weighted_average(pred_probs_1, pred_probs_2, weight_1=0.5, weight_2=0.5):
    return weight_1 * pred_probs_1 + weight_2 * pred_probs_2

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
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        eeg_test_data = torch.tensor(eeg_test_data, dtype=torch.float32)
        eye_test_data = torch.tensor(eye_test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        eeg_model = MLP_EEG(input_size=eeg_train_data.shape[1], hidden_size=64, output_size=3)
        eye_model = MLP_Eye(input_size=eye_train_data.shape[1], hidden_size=64, output_size=3)

        eeg_optimizer = optim.Adam(eeg_model.parameters(), lr=0.001)
        eye_optimizer = optim.Adam(eye_model.parameters(), lr=0.001)

        train(eeg_model, eeg_optimizer, eeg_train_data, train_labels, epochs=20)
        train(eye_model, eye_optimizer, eye_train_data, train_labels, epochs=20)

        eeg_model.eval()
        eye_model.eval()

        with torch.no_grad():
            eeg_probs = eeg_model(eeg_test_data).numpy()
            eye_probs = eye_model(eye_test_data).numpy()

            weight_1 = 0.5 
            weight_2 = 0.5 
            fused_probs = weighted_average(eeg_probs, eye_probs, weight_1, weight_2)

            final_preds = np.argmax(fused_probs, axis=1)

            accuracy = accuracy_score(test_labels.numpy(), final_preds)
            subject_accuracies.append(accuracy)
            print(f"Subject {subject_id} Accuracy: {accuracy * 100:.2f}%")

    average_accuracy = np.mean(subject_accuracies)
    print(f"Average Accuracy Across All Subjects: {average_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
