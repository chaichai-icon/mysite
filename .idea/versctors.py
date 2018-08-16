import torch
from torch.autograd import Variable

tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)  #相当于放在另一个篮子里面计算

print(tensor)
print(variable)

t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)
print(t_out)
print(v_out)

v_out.backward() #反向传递v_out=1/4*sum(var*var)
print(variable.grad,'\n',variable.data,'\n',variable.data.numpy())#输出反向传递









