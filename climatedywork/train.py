import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import codecs
from torchsummary import summary
 
 
 
f = codecs.open('/media/workdir/hujh/climatedywork/dmi.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
line = f.readline()   # 以行的形式进行读取文件
list1 = []
while line:
    a = line.split()
    b = a[2:3]   # 这是选取需要读取的位数
    list1.append(b)  # 将其添加在列表之中
    line = f.readline()
f.close()
 

dataset = np.array(list1).astype('float64')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value-min_value
dataset = np.array(list(map(lambda x: x/scalar, dataset))).squeeze() #将数据标准化到0~1之间
 
def create_dataset(dataset,look_back=60):
    look_back=60
    dataX, dataY=[],[]
    for i in range(len(dataset)-2*look_back):
        a=dataset[i:(i+look_back)]
        b = dataset[(i+look_back):(i+2*look_back)]

        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)
 
data_X, data_Y = create_dataset(dataset)
 
#划分训练集和测试集，1个作为测试集
train_size = int(len(data_X))-1
test_size = 1
 
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
 
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]
 
train_X = train_X.reshape(-1,60,1,1)
train_Y = train_Y.reshape(-1,60,1,1)
test_X = test_X.reshape(-1,1,60,1,1)
test_Y = test_X.reshape(-1,60,1,1)
"""
train_X = train_X[:,:,np.newaxis,np.newaxis]
train_Y = train_Y[:,:,np.newaxis,np.newaxis]
test_X = test_X[:,:,np.newaxis,np.newaxis]
test_Y = test_Y[:,:,np.newaxis,np.newaxis]"""


 
 
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
 
class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size=1,num_layers=2):
        super(lstm_reg,self).__init__()
 
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers)
        self.reg = nn.Linear(hidden_size,output_size)
 
    def forward(self,x):
        x, _ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x
 
 
net = lstm_reg(1,128)
net = net.double()
 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = net.to(device)
summary(lstm, input_size=(60,1))"""
 
for epoch in tqdm(range(1)):
    for i in range(train_X.shape[0]):
        var_x = train_x[i,...]
        var_y = train_y[i,...]
        #print(var_x.shape)
    
        out = net(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%10 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
        
torch.save(net.state_dict(), '/media/workdir/hujh/climatedywork/net_params.pkl')

net.eval()
with torch.no_grad():
    test_xx = test_x[0,...]
    
    test_xx = torch.squeeze(test_xx,0)
    print(test_xx.shape)
    test_out = net(test_xx)
    print(test_out)

    test_out_np = test_out.detach().cpu().numpy()
    #test_truth = np.concatenate((np.squeeze(test_X[0,...],axis = 0),np.squeeze(test_Y[0,...],axis= 0)),axis=0)
    #test_predict = np.concatenate((np.squeee(test_X[0,...],axis = 0),test_out_np),axis=0)
    print("test--------------------size:")
    xx = np.squeeze(test_X[0,...],axis = 0)
    print(xx.shape)
    test_truth = np.concatenate((xx,test_Y[0,...]),axis=0)
    test_predict = np.concatenate((xx,test_out_np),axis=0)
    test_truth = test_truth.squeeze()*scalar
    test_predict = test_predict.squeeze()*scalar
    plt.plot(test_predict, 'r', label='prediction')
    plt.plot(test_truth, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('/media/workdir/hujh/climatedywork/predict.png')
    plt.show()