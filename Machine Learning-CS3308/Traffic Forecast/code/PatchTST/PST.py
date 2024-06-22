import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch import Tensor
import matplotlib.pyplot as plt
import os

__all__ = ['PatchTST']


from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
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
            if not (seq_x[-1] == seq_x[0] + len(seq_x) - 1 and seq_y[-1] == seq_x[-1] + 1):
                i = i - self.pred_len - self.seq_len + 1
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

output_file = open('output/PatchTST.csv', 'w')

output_file.write('id,estimate_q\n')

seq_len = 24  
pred_len = 1  

k = 1

class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  
    revin = True  
    channel = 1  
    e_layers = 1
    n_heads = 128
    d_model = 128
    d_ff = 320
    dropout = 0.01
    fc_dropout = 0.01
    head_dropout = 0.01
    patch_len = 16
    stride = 8
    padding_patch = True
    decomposition = True
    kernel_size = 25
    affine = True
    subtract_last = True

dataset = TrafficDataset(file2, seq_len, pred_len)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

configs = Configs()
model = Model(configs)  
model = model.to('cuda')
train_loss = []
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00025)

device = torch.device('cuda')

model.train()
for epoch in range(600): 
    epoch_loss = 0
    for seq_x, seq_y in dataloader:
        seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output = model(seq_x.unsqueeze(2)) 
        loss = criterion(output, seq_y.unsqueeze(2))  
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
        output = model(seq_x.unsqueeze(2))
        predictions.append(output.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

for  prediction in predictions:
    estimate_q = int(round(prediction.item()))
    output_file.write(f"{k},{estimate_q}\n")
    k += 1
torch.save(model, 'output/PST.pth')
torch.save(model.state_dict(), 'output/PST_temp.pth')
output_file.close()

plt.figure()
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('output/PST.png')  

