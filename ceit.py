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
batch_size = 4
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((H, W)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_dataset_dir = 'c:/users/armin/desktop/animal'
train_dataset = torchvision.datasets.ImageFolder(train_dataset_dir, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#%%
def generate_pathes():
    pass

class MSA(nn.Module):
    def __init__ (self, N=512, h=3):
        super().__init__()
        self.n = N
        self.h = h
        self.q = nn.Linear(N,N)
        self.k = nn.Linear(N,N)
        self.v = nn.Linear(N,N)
        self.mha = nn.Linear(1536,512)
        self.LN = nn.LayerNorm([257, 512])
        self.FF = nn.Linear(N,N)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        k = k.transpose(1,2)
        for i in range(0,self.h):
            scores = torch.matmul(q, k) / np.sqrt(self.n)
            scores = F.softmax(scores, dim=-1)
            current_output = torch.matmul(scores, v)
            if i == 0:
                output = current_output
            else:
                output = torch.cat((output, current_output), dim=2)
        x_msa = self.mha(output)
        x = x + x_msa
        x = self.LN(x)  
        x_ff = self.FF(x)
        x = x + x_ff
        x = self.LN(x)
        # print(x.size());sys.exit()        
        return x
    
class FFN(nn.Module):
    def __init__ (self):
        super().__init__()
        self.gelu = nn.GELU()
    
    def forward(self):
        pass
    
class LeFF(nn.Module):
    def __init__ (self):
        super().__init__()
        self.e = 4
        self.gelu = nn.GELU()
        self.BN = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv2d = nn.Conv2d(16, 16, 1, stride=1)
        self.linear1 = nn.Linear(512,512*4)
        self.linear2 = nn.Linear(512*4,512)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.BN(x)
        x = self.gelu(x)
        x = x.view(4,16,16,2048)
        x = self.conv2d(x)
        x = self.BN2(x)
        x = self.gelu(x)
        x = x.view(4,256,2048)
        x = self.linear2(x)
        x = self.BN(x)
        x = self.gelu(x)
        return x
        
    
class I2T(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 2, 2) #(3,enriched_channel,kernel,stride) page5
        self.BN = nn.BatchNorm2d(32)
        self.MaxPool = nn.MaxPool2d(1, 2) #(kernel,stride) page5
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.BN(x)
        x = self.MaxPool(x)
        return x
 
class FFN(nn.Module):
    def __init__ (self, input_channel=3, output_channel=256):
        super().__init__()
        self.linear1 = nn.Linear(512,512, bias=True)
        self.linear2 = nn.Linear(512,512, bias=True)
        self.linear3 = nn.Linear(512,2, bias=True)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
    
class CeiT(nn.Module):
    def __init__ (self):
        super().__init__()
        # self.embeds = nn.Embedding(2, 1)
        self.i2t = I2T()
        self.position_embedding = torch.nn.parameter.Parameter(data=torch.rand(257,512), requires_grad=True)
        self.msa = MSA(N=512)
        self.lef = LeFF()
        self.ffn = FFN()
        
    def forward(self, x):
        x = self.i2t(x); 
        x = x.view(4,-1,32,4,4); 
        class_token = torch.randn(4, 1, 32, 4, 4)
        x = torch.cat((x, class_token), dim=1)
        x = x.view(4, 257, -1)
        x = x + self.position_embedding
        for i in range(0,12):
            x = self.msa(x)  
        x_split_class = x[:,0,:]
        x_split_class = x_split_class.unsqueeze(1)
        x_split_tokens = x[:,1:,:]
        x = self.lef(x_split_tokens)
        x = torch.cat((x, x_split_class), dim=1)
        x  = self.ffn(x_split_class)
        return x[:,0]


X = 3
P = 16 # patches size
N = int(H*W/(P**2)) # number of patches
input_img_stride = 4
C = 0 # latent embeddings size
maxpool_stride = 0
MSA_h = 0 # number of heads in MSA
MSA_N = 0
i2t_kernel_size = 3
i2t_stride = 3
i2t_maxpool_stride = 3

model = CeiT()
model.train()
smp = next(iter(train_dataloader))[0]

vv = model(smp)

#%%
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)
loss_func = nn.CrossEntropyLoss()

for epoch in range(2):
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(batch[0])[:,0]
        loss = loss_func(output, batch[1].to(torch.float32))
        total_loss = loss.item() + total_loss
        print(loss)
        loss.backward()
        optimizer.step()
    pritn(total_loss)
        # if i==10: sys.exit()
#%%