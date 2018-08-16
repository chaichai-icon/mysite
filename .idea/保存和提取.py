import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)

def save():
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    optimize=torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func=torch.nn.MSELoss()

    for t in range(100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimize.zero_grad()
        loss.backward()
        optimize.step()
    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),'net_params.pkl')
def restore_net():
    net2=torch.load('net.pkl')
def restore_param():
    net3=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))

save()

restore_net()

restore_param()
