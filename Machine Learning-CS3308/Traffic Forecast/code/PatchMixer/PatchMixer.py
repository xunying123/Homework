import torch
import torch.nn as nn
import torch.fft

import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    
# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
    
# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class PatchMixerLayer(nn.Module):
    def __init__(self,dim,a,kernel_size = 8):
        super().__init__()
        self.Resnet =  nn.Sequential(
            nn.Conv1d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim,a,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )
    def forward(self,x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                   # x: [batch * n_val, a, d_model]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)
    def forward(self, x):
        x = self.model(x)
        return x
class Backbone(nn.Module):
    def __init__(self, configs,revin = True, affine = True, subtract_last = False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        # if configs.a < 1 or configs.a > self.patch_num:
        #     configs.a = self.patch_num
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))
        self.W_P = nn.Linear(self.patch_size, self.d_model)  
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]  

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs , nvars, -1))                                       # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x

class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = self.create_samples()
        
#数据的处理待定
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

file1 = pd.read_csv('iiid.csv')

file3 = pd.read_csv('for_prediction.csv')

file2 = pd.read_csv('modified_file.csv')

output_file = open('PatchMixer-new.csv', 'w')
output_file.write('id,estimate_q\n')

seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

k = 1

'''
self.nvals = configs.enc_in
    self.lookback = configs.seq_len
    self.forecasting = configs.pred_len
    self.patch_size = configs.patch_len
    self.stride = configs.stride
    self.kernel_size = configs.mixer_kernel_size

    self.PatchMixer_blocks = nn.ModuleList([])
    self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
    self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
    # if configs.a < 1 or configs.a > self.patch_num:
    #     configs.a = self.patch_num
    self.a = self.patch_num
    self.d_model = configs.d_model
    self.dropout = configs.dropout
    self.head_dropout = configs.head_dropout
    self.depth = configs.e_layers
'''

class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  # 单通道，即每个探头一个时间序列
    revin = True  # 启用RevIN
    channel = 1  # 输入通道数量
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
    mixer_kernel_size = 8

dataset = TrafficDataset(file2, seq_len, pred_len)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

configs = Configs()
model = Model(configs)  # 创建模型实例

model = model.to('cuda:0')
train_loss = []
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device('cuda:0')

model.train()
for epoch in range(3000): 
    epoch_loss = 0
    for seq_x, seq_y in dataloader:
        seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output = model(seq_x.unsqueeze(2))  # 添加通道维度
        loss = criterion(output, seq_y.unsqueeze(2))  # 添加通道维度
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
torch.save(model, 'PatchMixer1.pth')
torch.save(model.state_dict(), 'PatchMixer_temp1.pth')
output_file.close()

