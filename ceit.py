import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
#%%
H = 256
W = 256

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((H, W)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_dataset_dir = 'c:/users/armin/desktop/animal'
train_dataset = torchvision.datasets.ImageFolder(train_dataset_dir, transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
#%%
X = 3
P = 16
C = 3
N = H*W/(P**2)
S = 4
#%%
def generate_pathes():
    pass

class MSA(nn.Module):
    def __init__ (self):
        super().__init__()
        self.q = nn.Linear(256,256)
        self.k = nn.Linear(256,256)
        self.v = nn.Linear(256,256)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        k = k.transpose(1,2)
        scores = torch.matmul(q, k) / np.sqrt(256)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output
    
class FFN(nn.Module):
    def __init__ (self):
        super().__init__()
        self.gelu = nn.GELU()
    
    def forward(self):
        pass
    
class LeFF(nn.Module):
    def __init__ (self):
        super().__init__()
        self.gelu = nn.GELU()
        self.BN = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv2d = nn.Conv2d(16, 16, 1, stride=1)
        self.to_h = nn.Linear(256,1024)
        self.to_h2 = nn.Linear(1024,256)
        
        
    def forward(self, x):
        x = self.to_h(x)
        
        x = self.BN(x)
        x = self.gelu(x)
        
        x = x.view(4,16,16,1024)
        x = self.conv2d(x)
        
        x = self.BN2(x)
        x = self.gelu(x)
        
        x = x.view(4,256,1024)
        x = self.to_h2(x)
        return x
        
    
class I2T(nn.Module):
    def __init__ (self, input_channel=3, output_channel=256):
        super().__init__()
        self.conv2d = nn.Conv2d(input_channel, output_channel, 3, stride=4)
        self.BN = nn.BatchNorm2d(output_channel)
        self.MaxPool = nn.MaxPool2d(3, stride=4)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.BN(x)
        x = self.MaxPool(x)
        return x
 
class FFN(nn.Module):
    def __init__ (self, input_channel=3, output_channel=256):
        super().__init__()
        self.to_h1 = nn.Linear(256,256)
        self.to_h2 = nn.Linear(256,256)
        self.to_h3 = nn.Linear(256,2)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.to_h1(x)
        x = self.gelu(x)
        x = self.to_h2(x)
        x = self.to_h3(x)
        return x
    
    
class CeiT(nn.Module):
    def __init__ (self, input_channel=3, output_channel=256):
        super().__init__()
        self.embeds = nn.Embedding(2, 1)
        self.i2t = I2T(input_channel=3, output_channel=256)
        self.position_embedding = torch.nn.parameter.Parameter(data=torch.rand(257,256), requires_grad=True)
        self.msa = MSA()
        self.lef = LeFF()
        self.ffn = FFN()
    def forward(self, x):
        x = self.i2t(x)
        
        class_token = torch.randn(4, 1, 16, 16)
        x = torch.cat((x, class_token), dim=1)
        
        x = x.view(4, 257, -1)
        
        x = x + self.position_embedding # 4,33,256
        x = self.msa(x)
        x_withoutclass = x[:,0,:]
        x_withclass = x[:,1:,:]
        x_withoutclass = x_withoutclass.unsqueeze(1)
        # x_withclass = x_withclass.unsqueeze(1)
        x = self.lef(x_withclass)
        x = torch.cat((x, x_withoutclass), dim=1)
        x  = self.ffn(x_withoutclass)
        return x
    
model = CeiT()
model.train()
smp = next(iter(train_data_loader))[0]

vv = model(smp)

#%%
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()

for epoch in range(2):
    total_loss = 0
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        output = model(batch[0])[:,0]
        # print(output[:,0])
        # print((batch[1].to(torch.float32)))2
        # output = output.squeeze(1)
        # print(output)
        # output2 = torch.argmax(output,dim=1)
        loss = loss_func(output, batch[1])
        print(loss)
        # loss = loss_func(output.to(torch.float32), batch[1].to(torch.float32))
        loss.backward()
        optimizer.step()
        if i==300: sys.exit()
    #     src = batch[0].to(device)
    #     src_mask = batch[1].to(device)
    #     trg = batch[2].to(device)
    #     trg_mask = batch[3].to(device)
    #     optimizer.zero_grad()
    #     preds = model(src, trg[:,:-1], src_mask, trg_mask[:,:-1])
    #     ys = (trg[:,1:]).to(device)
    #     ys = ys.contiguous().view(-1)
    #     pred = preds.view(-1, preds.size(-1))
    #     loss = loss_func(pred, ys)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss = loss.item() + total_loss
    #     # if i == 100: sys.exit()
    #     if (i % 500) == 0:
    #         print(loss)
    # total_loss_list.append(total_loss)
    # print(total_loss)
#%%
next(iter(train_data_loader))[1].size()
output2.size()