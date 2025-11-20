import torch

class LinearModel(torch.nn.module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
    #class torch.nn.Linear(in_features, out_features, bias=True)
    #in_features: size of each input sample
    #out_features: size of each output sample
    #whether or not to add a bias(b)
model = LinearModel#callable model

critertion = torch.nn.MSELoss(size_average=False)#loss

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)



 
