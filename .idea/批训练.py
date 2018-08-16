import torch
import torch.utils.data as Data

torch.manual_seed(1)
BATCH_SIZE=5
x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)  #定义一个数据库
#torch_dataset = Data.TensorDataset(x,y)  #定义一个数据库


loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,    #loader要不要打乱数据排序
    num_workers=2,
)

for t in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print('Epoch',t,'|step:',step,'|batch x:',batch_x.numpy(),'|batch y:',batch_y.numpy())

