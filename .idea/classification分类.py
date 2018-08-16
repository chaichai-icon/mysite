import torch
import numpy as np



#abs
data=[[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data=np.array(data)
# print(
#     '\nabs',
#     '\nnumpy',np.abs(data),
#     '\ntorch',torch.abs(tensor),
# )
#矩阵的乘法
# print(
#     '\nnumpy:',data.dot(data),
#     '\ntorch',tensor.dot(tensor)#这个不对，为什么？
# )
#mean sin cos abs  matmul/mm矩阵运算(显示错误)




np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tersor2arry=torch_data.numpy()
#print(
#     '\nnumpy:',np_data,
#     '\ntorch:',torch_data,
#     '\ntersor2arry:', tersor2arry,
# )