import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
from torch import nn
from tqdm import tqdm
import codecs
#from torchsummary import summary
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

f = codecs.open(r'C:\Users\alexanderhu\Desktop\climatedywork\dmi.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
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
scalar_northarea = max_value-min_value
dataset = np.array(list(map(lambda x: x/scalar_northarea , dataset))).squeeze() #将数据标准化到0~1之间
### input must match the target seq it has 408 times begin 1979.1-2012.11
dataset = dataset[0:408]
# we must set the dataset size >1*1 so we make it become 4*4
dataset_area = np.ones(6528)
dataset_area = dataset_area.reshape(408,4,4)

# reduce the for loop is a beautifal work
for i in range(dataset_area.shape[0]):
    dataset_area[i,...] = dataset_area[i,...]*dataset[i]

###########

path = r'C:\Users\alexanderhu\Desktop\climatedywork\mslp.mon.mean.r2.nc'
fileobj = nc.Dataset(r'C:\Users\alexanderhu\Desktop\climatedywork\mslp.mon.mean.r2.nc' ,'r')
print("ok")
##print(fileobj)
#print (fileobj.variables)
lat = fileobj.variables['lat'][:]
lon = fileobj.variables['lon'][:]
slp = fileobj.variables['mslp'][:,:,:]
time = fileobj.variables['time']
times = nc.num2date(time[:],time.units)
#print(len(times))
slp[slp==32766]=0.0
slp = np.array(slp)
slp_max = slp.max()
slp_min = 0.0
scalar_slp = slp_max - slp_min
#normalization
slp = slp/(scalar_slp+0.0)
slp = np.clip(slp,0,1)
slp = slp.transpose(0,2,1)

#划分训练集和测试集，1个作为测试集
train_size = int(len(slp))-4
test_size = 4
 
train_X = dataset_area[:train_size]
train_Y = slp[:train_size]
 
test_X = dataset_area[train_size:]
test_Y = slp[train_size:]

device = torch.device('cuda:0')
train_x = torch.from_numpy(train_X).to(device)
train_y = torch.from_numpy(train_Y).to(device)
print("the early datasets:")
print(train_x.shape)
print(train_y.shape)
test_x = torch.from_numpy(test_X).to(device)
test_y = test_Y 

#define the transposed2d model
# the input size is 1*4*4 to become a 1*144*73
class deconv2d(nn.Module):
    def __init__(self):
        super(deconv2d,self).__init__()
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=1,out_channels =8,kernel_size=4,stride=2,padding=1),
                                     nn.LeakyReLU(inplace = True),
                                     nn.ConvTranspose2d(in_channels=8,out_channels =16,kernel_size=4,stride=2,padding=1),
                                     nn.LeakyReLU(inplace = True),
                                     nn.ConvTranspose2d(in_channels=16,out_channels =32,kernel_size=4,stride=2,padding=1),
                                     nn.LeakyReLU(inplace = True)) ### size = 32*32*32
        self.unmaxpool = nn.Sequential(nn.ConvTranspose2d(in_channels=32,out_channels =32,kernel_size=4,stride=2,padding=1),
                                       nn.LeakyReLU(inplace = True))
        self.deconv2 = nn.ConvTranspose2d(in_channels=32,out_channels =1,kernel_size=(18,10),stride=(2,1)) ### 不想算了，直接变成size = 1*144*73
    def forward(self,x):
        x = self.deconv1(x)
        x = self.unmaxpool(x)
        x = self.deconv2(x)
        return x


model = deconv2d()
model = model.double()
model.to(device)
lr=1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-3)
for epoch in tqdm(range(100)):
    if epoch//10 ==0:
            lr = lr *0.7
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for i in range(train_X.shape[0]):
        var_x = train_x[i,...]
        var_y = train_y[i,...]
        var_x  = var_x.unsqueeze(0).unsqueeze(0)
        var_y  = var_y.unsqueeze(0).unsqueeze(0)
        #print("the x input train data size is:")
        #print(var_x.shape)
        #print("the y input train data size is:")
        #print(var_y.shape)
    
        out = model(var_x)
        #print("the out size is :")
        #print(out.shape)
        out = out.double()
        var_y =var_y.double()
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%10 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
        
#torch.save(model.state_dict(), '/media/workdir/hujh/climatedywork/deconv_net_params.pkl')
#### test 
model.eval()
with torch.no_grad():
    test_xx = test_x[0,...]
    #test_xx = train_x[40,...]
    test_xx = test_xx.unsqueeze(0).unsqueeze(0)
    print(test_xx.shape)
    test_out = model(test_xx)
    #print(test_out)

    test_out_np = test_out.detach().cpu().numpy()
    #test_truth = np.concatenate((np.squeeze(test_X[0,...],axis = 0),np.squeeze(test_Y[0,...],axis= 0)),axis=0)
    #test_predict = np.concatenate((np.squeee(test_X[0,...],axis = 0),test_out_np),axis=0)
    #print("test--------------------size:")
    #xx = test_X[0,...]
    #print(xx.shape)
    
    #test_truth = test_Y[0,...]
    test_truth = test_Y[0,...]
    test_predict = test_out_np[0,0,...]
    test_truth = test_truth.squeeze()*scalar_slp
    test_predict = test_predict.squeeze()*scalar_slp
    test_truth = np.transpose(test_truth,(1,0))
    test_predict =np.transpose(test_predict,(1,0))
    print(test_truth.shape)
    print(test_predict.shape)
    """
    plt.plot(test_predict, 'r', label='prediction')
    plt.plot(test_truth, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('/media/workdir/hujh/climatedywork/predict_area.png')
    plt.show()"""
##########################################   vislation ##########################
fig = plt.figure(figsize=(8, 10))
#set colormap
#levels = MaxNLocator(nbins=16).tick_values(-2.0, 2.0) #设置等值线值
cmap   = plt.get_cmap('bwr')  #选择coloarmap
#norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True) #标准化coloarmap
#画第一个子图
# Label axes of a Plate Carree projection with a central longitude of 180:
ax1 = plt.subplot(211, projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_global() #使得轴域（Axes即两条坐标轴围城的区域）适应地图的大小
ax1.coastlines() #画出海岸线
ax1.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree()) #标注坐标轴
ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True) #zero_direction_label=False 用来设置经度的0度加不加E和W
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter) #标注坐标轴
#画等值线
ax1.set_title("target truth")
plt.contourf(lon,lat,test_truth,15, cmap=cmap,transform=ccrs.PlateCarree())
plt.colorbar(orientation='horizontal')

# Label axes of a Mercator projection without degree symbols in the labels
# and formatting labels to include 1 decimal place:
ax2 = plt.subplot(212, projection=ccrs.PlateCarree(central_longitude=180))
ax2.set_global()
ax2.coastlines()
ax2.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True) #zero_direction_label=False 用来设置经度的0度加不加E和W
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.set_title("prediction")

#画等值线
plt.contourf(lon,lat,test_predict,15, cmap=cmap,transform=ccrs.PlateCarree())
plt.colorbar(ax=ax2,orientation='horizontal')
fig.tight_layout()
plt.savefig(r"C:\Users\alexanderhu\Desktop\climatedywork\compare.png") 
