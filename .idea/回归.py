import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

#造数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)    #一维变二维
y=x.pow(2)+0.2*torch.rand(x.size())

x,y=Variable(x),Variable(y)

# #打印出来
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


#搭建神经网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()

        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)  #输出层的输出
        return x

net=Net(1,10,1)
print(net)


plt.ion()   #实时打印
plt.show()

#优化神经网络
optimize=torch.optim.SGD(net.parameters(),lr=0.5)  #lr学习速率
loss_func=torch.nn.MSELoss()

#训练
for t in range(100):
    prediction=net(x)

    #计算误差
    loss=loss_func(prediction,y)  #算误差，计算值和真实值

    #优化
    optimize.zero_grad()  #
    loss.backward()   #反向传递
    optimize.step()   #优化梯度

    #可视化
    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'loss=%.5f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()




























