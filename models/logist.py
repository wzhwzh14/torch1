import torch
import numpy as np
import matplotlib.pyplot as plt
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0.0],[0.0],[1.0]])
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    #class torch.nn.Linear(in_features, out_features, bias=True)
    #in_features: size of each input sample
    #out_features: size of each output sample
    #whether or not to add a bias(b)
model = LogisticRegressionModel()#callable model

critertion = torch.nn.BCELoss(size_average=False)#loss

optimizer = torch.optim.Rprop(model.parameters(),lr=0.04)

for epoch in range(100):
    #forward pass
    y_pred = model(x_data)
    
    #compute loss
    loss = critertion(y_pred,y_data)
    print("epoch:",epoch,"loss:",loss.item())
    
    #zero gradients
    optimizer.zero_grad()
    
    #backward pass
    loss.backward()
    
    #update weights
    optimizer.step()
    #1. y hat
    #2. loss function
    #3. backward pass
    #4. update weights

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("predict:",y_test.data)

x=np.linspace(0,10,200)
x_t=torch.tensor(x,dtype=torch.float32).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r',linestyle='--')
plt.xlabel('Hours')
plt.ylabel('Pass Probability')
plt.grid()
plt.show()