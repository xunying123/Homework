import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

dataset_path = "dataset"

time_steps = 3  
features = 310  
hidden_size = 128  
num_classes = 3  
num_epochs = 50 
batch_size = 32
learning_rate = 0.001

subjects_data = []
subjects_labels = []

for subject_folder in sorted(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject_folder)
    data = np.load(os.path.join(subject_path, "data.npy"))  
    labels = np.load(os.path.join(subject_path, "label.npy"))  

    data = data.reshape(data.shape[0], -1)
    
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data_std[data_std == 0] = 1  
    data = (data - data_mean) / data_std

    sequences = []
    sequence_labels = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i:i + time_steps]) 
        sequence_labels.append(labels[i + time_steps - 1])  

    subjects_data.append(np.array(sequences))  
    subjects_labels.append(np.array(sequence_labels))

label_encoder = LabelEncoder()
for i in range(len(subjects_labels)):
    subjects_labels[i] = label_encoder.fit_transform(subjects_labels[i])

class EmotionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.fc(out[:, -1, :])  
        return out

results = []

for test_subject in range(len(subjects_data)):
    x_test = subjects_data[test_subject]
    y_test = subjects_labels[test_subject]
    x_train = np.vstack([subjects_data[i] for i in range(len(subjects_data)) if i != test_subject])
    y_train = np.hstack([subjects_labels[i] for i in range(len(subjects_data)) if i != test_subject])

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmotionRNN(features, hidden_size, num_classes)
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

print(f'Mean = {mean_accuracy:.4f}, Std = {std_accuracy:.4f}')
