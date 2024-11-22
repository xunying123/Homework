import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

layout = [
    ['-', '-', 'AF3', 'FP1', 'FPZ', 'FP2', 'AF4', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]

channelID2str = {
    1: 'FP1', 2: 'FPZ', 3: 'FP2', 4: 'AF3', 5: 'AF4',
    6: 'F7', 7: 'F5', 8: 'F3', 9: 'F1', 10: 'FZ', 11: 'F2', 12: 'F4', 13: 'F6', 14: 'F8',
    15: 'FT7', 16: 'FC5', 17: 'FC3', 18: 'FC1', 19: 'FCZ', 20: 'FC2', 21: 'FC4', 22: 'FC6', 23: 'FT8',
    24: 'T7', 25: 'C5', 26: 'C3', 27: 'C1', 28: 'CZ', 29: 'C2', 30: 'C4', 31: 'C6', 32: 'T8',
    33: 'TP7', 34: 'CP5', 35: 'CP3', 36: 'CP1', 37: 'CPZ', 38: 'CP2', 39: 'CP4', 40: 'CP6', 41: 'TP8',
    42: 'P7', 43: 'P5', 44: 'P3', 45: 'P1', 46: 'PZ', 47: 'P2', 48: 'P4', 49: 'P6', 50: 'P8',
    51: 'PO7', 52: 'PO5', 53: 'PO3', 54: 'POZ', 55: 'PO4', 56: 'PO6', 57: 'PO8', 58: 'CB1', 59: 'O1', 60: 'OZ', 61: 'O2', 62: 'CB2'
}

time_steps = 3 
features = (8, 9, 5)  
hidden_size = 128
num_classes = 3
num_epochs = 50
batch_size = 32
learning_rate = 0.001

subjects_data = []
subjects_labels = []

dataset_path = "dataset"
for subject_folder in sorted(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject_folder)
    data = np.load(os.path.join(subject_path, "data.npy"))  
    labels = np.load(os.path.join(subject_path, "label.npy"))  

    data_mean = np.mean(data, axis=0, keepdims=True) 
    data_std = np.std(data, axis=0, keepdims=True) 
    data_std[data_std == 0] = 1  
    data = (data - data_mean) / data_std  

    filled_data = []
    for sample in data:
        sample_filled = np.zeros((8, 9, 5)) 
        for row_idx, row in enumerate(layout):
            for col_idx, channel_name in enumerate(row):
                if channel_name == '-':
                    sample_filled[row_idx, col_idx] = np.zeros(5)
                else:
                    channel_id = [key for key, val in channelID2str.items() if val == channel_name]
                    if channel_id:
                        sample_filled[row_idx, col_idx] = sample[channel_id[0] - 1]  
        filled_data.append(sample_filled)

    filled_data = np.array(filled_data)
    sequences = []
    sequence_labels = []
    for i in range(len(filled_data) - time_steps + 1):
        sequences.append(filled_data[i:i + time_steps])
        sequence_labels.append(labels[i + time_steps - 1])

    subjects_data.append(np.array(sequences))  
    subjects_labels.append(np.array(sequence_labels))

label_encoder = LabelEncoder()
for i in range(len(subjects_labels)):
    subjects_labels[i] = label_encoder.fit_transform(subjects_labels[i])

class EmotionCNN(nn.Module):
    def __init__(self, time_steps, input_channels, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(64 * (time_steps // 2) * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

results = []

for test_subject in range(len(subjects_data)):
    x_test = subjects_data[test_subject]
    y_test = subjects_labels[test_subject]
    x_train = np.vstack([subjects_data[i] for i in range(len(subjects_data)) if i != test_subject])
    y_train = np.hstack([subjects_labels[i] for i in range(len(subjects_data)) if i != test_subject])

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).permute(0, 4, 1, 2, 3), 
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32).permute(0, 4, 1, 2, 3),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmotionCNN(time_steps=time_steps, input_channels=5, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)

    all_preds = torch.cat(all_preds).numpy()
    accuracy = accuracy_score(y_test, all_preds)
    results.append(accuracy)
    print(accuracy)

mean_accuracy = np.mean(results)
std_accuracy = np.std(results)
print(f'Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}')
