import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader 
import torch.nn.functional as F
import torch.optim as optim     

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,128)
        self.fc4 = torch.nn.Linear(128,64)
        self.fc5 = torch.nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    model.train()
    runningloss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        runningloss+=loss.item()
        if batch_idx %300 ==299:
            print('[%d,%5d] loss: %.3f'%(epoch+1,batch_idx+1,runningloss/300
            ))
            runningloss=0.0

def test():
    total = 0
    correct =0
    with torch.no_grad():
        for data,target in test_loader:
            outputs = model(data)
            _,predicted = torch.max(outputs.data,1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %%' % (100*correct/total
    ))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()