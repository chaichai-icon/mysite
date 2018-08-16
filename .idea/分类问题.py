import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

#造数据
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

x,y=Variable(x),Variable(y)

# #打印出来
# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
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

net=Net(2,10,2)
print(net)

#第二种方法
net2=torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(net2)

plt.ion()   #实时打印
plt.show()

#优化神经网络
optimize=torch.optim.SGD(net.parameters(),lr=0.002)  #lr学习速率
loss_func=torch.nn.CrossEntropyLoss()   #分类问题

#训练
for t in range(100):
    out=net(x)

    #计算误差
    loss=loss_func(out,y)  #算误差，计算值和真实值

    #优化
    optimize.zero_grad()  #清除上一次结果
    loss.backward()   #反向传递
    optimize.step()   #优化梯度

    #可视化
    if t%2==0:  #每两步出图一次
        plt.cla()
        prediction=torch.max(F.softmax(out),1)[1] #正真的输出即概率
        pred_y=prediction.data.numpy().squeeze()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0)
        accuracy=sum(pred_y==target_y)/200
        plt.text(1.5,-4,'loss=%.5f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
