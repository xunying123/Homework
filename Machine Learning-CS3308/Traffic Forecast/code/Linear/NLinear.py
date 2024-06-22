import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x 
    
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

output_file = open('output/NLinear.csv', 'w')

output_file.write('id,estimate_q\n')

seq_len = 24  
pred_len = 1  

k = 1
class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  

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
        output = model(seq_x.unsqueeze(-1)) 
        loss = criterion(output, seq_y.unsqueeze(-1))  
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
        output = model(seq_x.unsqueeze(-1))
        predictions.append(output.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

for  prediction in predictions:
    estimate_q = int(round(prediction.item()))
    output_file.write(f"{k},{estimate_q}\n")
    k += 1
torch.save(model, 'output/NLinear.pth')

output_file.close()

plt.figure()
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('output/NLinear.png')  