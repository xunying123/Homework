import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score

class MLP(nn.Module):
    def __init__(self, input_size=343, num_classes=3):
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

learning_rate = 0.001
num_epochs = 100
batch_size = 64

subject_accuracies = []
for subject_id in range(1, 13):
    eeg_train_data = np.load(f'dataset/{subject_id}/train_data_eeg.npy')
    eye_train_data = np.load(f'dataset/{subject_id}/train_data_eye.npy')
    eeg_test_data = np.load(f'dataset/{subject_id}/test_data_eeg.npy')
    eye_test_data = np.load(f'dataset/{subject_id}/test_data_eye.npy')
    train_labels = np.load(f'dataset/{subject_id}/train_label.npy')
    test_labels = np.load(f'dataset/{subject_id}/test_label.npy')

    scaler_eeg = StandardScaler()
    scaler_eye = StandardScaler()
    eeg_train_data_normalized = scaler_eeg.fit_transform(eeg_train_data)
    eye_train_data_normalized = scaler_eye.fit_transform(eye_train_data)
    eeg_test_data_normalized = scaler_eeg.transform(eeg_test_data)
    eye_test_data_normalized = scaler_eye.transform(eye_test_data)

    X_train = np.concatenate((eeg_train_data_normalized, eye_train_data_normalized), axis=1)
    X_test = np.concatenate((eeg_test_data_normalized, eye_test_data_normalized), axis=1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

    input_size = X_train.shape[1]
    output_size = len(np.unique(train_labels)) 
    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
 
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Subject {subject_id}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor, predicted)
        subject_accuracies.append(accuracy)
        print(f"Subject {subject_id} Accuracy: {accuracy * 100:.2f}%")

average_accuracy = np.mean(subject_accuracies)
print(f"Average Accuracy Across All Subjects: {average_accuracy * 100:.2f}%")
