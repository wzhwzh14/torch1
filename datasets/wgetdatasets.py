import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
train_set = torchvision.datasets.MNIST(root='../datasets/mnist', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='../datasets/mnist', train=False, download=True)

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32, skiprows=1)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset('D:/code cook/torch1/datasets/diabetes.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model=Model()
loss_history = []
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
if __name__ == '__main__':
    for epoch in range(50):
        running_loss = 0.0  # 用于累计当前epoch的总损失
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            
            # 前向传播
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()

        # --- 新增代码：计算并存储平均损失 ---
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print('Epoch [{}/{}], Average Loss: {:.6f}'.format(epoch+1, 100, epoch_loss))

    print('Finished Training')

    # --- 新增代码：绘制损失变化图 ---
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True) # 添加网格线，方便查看
    plt.show()
