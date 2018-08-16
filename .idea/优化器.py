import torch
from torch.autograd import  Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

LR=0.01
BATCH_SIZE=32
EPOCH=12


x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(),y.numpy())
# plt.show()


torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()

nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD     =torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.Momentum(net_Momentum(),lr=LR,momentum=0.8)
opt_RMSprop =torch.optim.RMSprop




# optimize=torch.optim.SGD()
# torch.optim.

