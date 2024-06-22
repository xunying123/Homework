import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Invertible import RevIN
import matplotlib.pyplot as plt
import os

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None
        self.individual = configs.individual

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)

class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = self.create_samples()
        
    def create_samples(self):
        samples = []
        values = self.data['q'].values
        for i in range(0, len(values) - self.seq_len - self.pred_len + 1, self.seq_len + self.pred_len):
            seq_x = values[i:i+self.seq_len]
            seq_y = values[i+self.seq_len:i+self.seq_len+self.pred_len]
            samples.append((seq_x, seq_y))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.samples[idx]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


file3 = pd.read_csv('data/temp/for_test.csv')

file2 = pd.read_csv('data/temp/for_train.csv')

os.makedirs('output', exist_ok=True)

output_file = open('output/RLinear.csv', 'w')

output_file.write('id,estimate_q\n')

seq_len = 24  
pred_len = 1  

k = 1
class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  
    drop = 0.2  
    rev = True 
    channel = 1  

dataset = TrafficDataset(file2, seq_len, pred_len)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

configs = Configs()
model = Model(configs)
model = model.to('cuda')
train_loss = []
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

device = torch.device('cuda')

model.train()
for epoch in range(500): 
    epoch_loss = 0
    for seq_x, seq_y in dataloader:
        seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output, loss = model(seq_x.unsqueeze(-1), seq_y.unsqueeze(-1)) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
    train_loss.append(epoch_loss / len(dataloader))
    
predict_dataset = TrafficDataset(file3, seq_len, 0)
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for seq_x, seq_y in predict_loader:
        seq_x = seq_x.to(device)
        output, _ = model(seq_x.unsqueeze(-1), seq_x.unsqueeze(-1))
        predictions.append(output.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

for  prediction in predictions:
    estimate_q = int(round(prediction.item()))
    output_file.write(f"{k},{estimate_q}\n")
    k += 1
torch.save(model, 'output/RLinear.pth')

output_file.close()

plt.figure()
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('output/RLinear.png')  

