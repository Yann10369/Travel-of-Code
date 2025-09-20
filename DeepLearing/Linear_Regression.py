import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

true_W = torch.tensor([3.2,2.2],dtype=torch.float32)
true_b = torch.tensor([0.5],dtype=torch.float32)
num_samples = 1000
input_dim =true_W.numel()
X=torch.randn(num_samples,input_dim)
noise = torch.randn(num_samples)*0.01  # 修改为1D张量
y=torch.matmul(X,true_W)+true_b+noise
y=y.unsqueeze(1)  # 确保y是2D张量 (1000, 1)

train_size= int(num_samples*0.8)
x_train,x_test=X[:train_size],X[train_size:]
y_train,y_test=y[:train_size],y[train_size:]
train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

model=nn.Sequential(nn.Linear(input_dim,1))

loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0099999999)
num_epochs=100
train_losses=[]
test_losses=[]
learning_rates=[]
for epoch in range(num_epochs):
    model.train()
    train_loss=0.0
    for train_x,train_y in train_loader:
        optimizer.zero_grad()#optimizer equal to zero
        y_hat=model(train_x)#predict value of y
        loss=loss_function(y_hat,train_y)#compute the loss
        loss.backward()#compute the gredient
        optimizer.step()#update the W
        train_loss+=loss.item()*train_x.size(0)
    train_losses.append(train_loss/len(train_loader.dataset))
    model.eval()
    test_loss=0.0
    with torch.no_grad():
        for test_x,test_y in test_loader:
            y_hat=model(test_x)
            loss=loss_function(y_hat,test_y)
            test_loss+=loss.item()*test_x.size(0)
    test_losses.append(test_loss/len(test_loader.dataset))
# 获取训练后的参数
with torch.no_grad():
    learned_W = model[0].weight.data.squeeze()
    learned_b = model[0].bias.data.item()

print(f"\n原始参数: W = {true_W.numpy()}, b = {true_b.item():.3f}")
print(f"学习参数: W = {learned_W.numpy()}, b = {learned_b:.3f}")
# 绘制学习率变化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.tight_layout()
plt.show()