import torch
import glob
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import linecache
from skimage import io
from skimage.transform import resize
import time
###############提取标注数据中的坐标，用1x22的向量表示（x1,y1,x2,y2,...,x11,y11）###########
def getnum(file):
    Num = np.zeros((11, 2))
    Num1 = np.zeros((1, 22))
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for I in range(1, 12):
        a = np.zeros((10))
        s = linecache.getline(file, I)
        index = 0
        num = 0
        j = 0
        for i in range(0, 10):
            if s[i] in number:
                a[i] = int(s[i])
                j = j + 1
            else:
                index = index + 1
                if index == 1:
                    aa = bb = j
                    for k in range(0, j):
                        num = num + a[k] * (10 ** (bb - 1))
                        bb = bb - 1
                else:
                    aaa = j - aa
                    for k in range(aa + 1, j + 1):
                        num = num + a[k] * (10 ** (aaa - 1))
                        aaa = aaa - 1
                Num[I - 1][index - 1] = num
                num = 0
                if index < 2:
                    continue
                else:
                    break
    for i in range(11):
        Num1[0,i]=Num[i][0]
        Num1[0,i+11] = Num[i][1]
    Num.resize((1,22))
    return Num

###################数据加载##########
def get_data(path, split, num):                   #num为数据集中样本的个数
    print('start loading ' + split + ' data...')
    X = np.zeros((num, 1, 224, 224), np.float)    #图片数据，将所有图片resize为224X224大小
    Y = np.zeros((num, 22), np.float)             #坐标数据
    Z = np.zeros((num, 2), np.float)              #每张图片的变化比例，用于将resize后得到的坐标变回原图中的坐标
    #如果之前已经加载过数据，则直接读取数据文件，节省时间
    if os.path.exists('X_Train1.npy') and os.path.exists('X_Test1.npy'):
        if split=='train/data':
            if os.path.exists('X_Train1.npy'):
               print("get Train data")
               X = np.load('X_Train1.npy')
               Y = np.load('Y_Train1.npy')
               Z = np.load('Z_Train1.npy')
        if split == 'test/data':
            if os.path.exists('X_Test1.npy'):
               print("get Test data")
               X = np.load('X_Test1.npy')
               Y = np.load('Y_Test1.npy')
               Z = np.load('Z_Test1.npy')
    else:     #加载数据
           if split == 'test/data' or 'train/data':
              for ind, file in enumerate(glob.glob(path + split + '/*.jpg')):
                f = file.replace('.jpg' , '.txt')
                img = io.imread(file)
                long=np.max(img) - np.min(img)
                min=np.min(img)
                img = (img - min) / long
                x,y=img.shape
                Z[ind, 0] = 224 / x
                Z[ind, 1] = 224 / y
                img = resize(img, (224,224))
                fea = img.astype(np.float)
                X[ind, 0, :] = fea
                Y[ind, :] = getnum(f)
           #保存数据，避免重复读取
           if(split=='train/data'):
             np.save('X_Train1.npy', X)
             np.save('Y_Train1.npy', Y)
             np.save('Z_Train1.npy', Z)
           elif(split=='test/data'):
             np.save('X_Test1.npy', X)
             np.save('Y_Test1.npy', Y)
             np.save('Z_Test1.npy', Z)
    print('finish loading ' + split + ' data...')
    #将X、Y数据从numpy转为tensor形式
    X=torch.from_numpy(X)
    X=X.float()
    Y = torch.from_numpy(Y)
    Y = Y.float()
    return X, Y, Z


####################模型定义（卷积神经网络，CNN）##########
def define_model():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 5, stride=2)    #卷积层C1，5x5、步长为2的卷积核，feature_map=16
            self.pool1 = nn.MaxPool2d(2, 2)               #2x2池化层
            self.conv2 = nn.Conv2d(16, 32, 4,stride=2)    #卷积层C2，4x4、步长为2的卷积核，feature_map=32
            self.pool2 = nn.MaxPool2d(2, 2)               #2x2池化层
            self.conv3 = nn.Conv2d(32, 64, 3)             #卷积层C3，3x3、步长为1的卷积核，feature_map=64
            self.pool3 = nn.MaxPool2d(2, 2)               #2x2池化层
            self.conv4 = nn.Conv2d(64, 64, 3)             #卷积层C4，3x3、步长为1的卷积核，feature_map=64
            # 三个全连接层
            self.fc1 = nn.Linear(64 * 3 * 3, 120)
            self.fc15 = nn.Linear(120, 84)
            self.fc2 = nn.Linear(84, 22)      #最后输出为1x22的向量

        def forward(self, x):
            #卷积层采用sigmoid作为激活函数，池化层无激活函数
            x = self.pool1(torch.sigmoid(self.conv1(x)))
            x = self.pool2(torch.sigmoid(self.conv2(x)))
            x = self.pool3(torch.sigmoid(self.conv3(x)))
            x=torch.sigmoid(self.conv4(x))
            #全连接层采用relu作为激活函数
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc15(x))
            x = self.fc2(x)
            return x
    net = Net()
    return net


################损失函数定义（MSELOSS）#########
def define_loss():
    Loss = torch.nn.MSELoss()
    return Loss


##############优化器定义#############
def define_optimizer(learning_rate):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)
    return optimizer

###################模型训练#########
def train(x, y, net, Loss, optimizer,Z,path):
    print('start training:')
    d=-2
    loss_1=9999999
    for t in range(30):
        y_pred = net(x)# 前向传播：通过像模型输入x计算预测的y
        #将坐标变回原图中对应坐标
        for j in range(149):
            for i in range(22):
                if i % 2 == 0:
                    y_pred[j, i] /= Z[j, 0]
                else:
                    y_pred[j, i] /= Z[j, 1]
        loss = Loss(y_pred, y)  # 计算loss
        print("第{}次, MSEloss为 {}".format(t + 1, loss.item()))
        #如果训练过程中过了极小值点，则降低学习率从前一步的参数重新训练
        if loss>loss_1:
            print('turn back')
            net.load_state_dict(torch.load(path+'cifar_model.pth'))
            d=d-1
            optimizer = define_optimizer(10 ** d)
        else:
            torch.save(net.state_dict(),path+'cifar_model.pth')
            loss_1 = loss
            optimizer.zero_grad()  # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零
            loss.backward()  # 反向传播：根据模型的参数计算loss的梯度
            optimizer.step()  # 调用Optimizer的step函数使它所有参数更新

    y_pred = net(x.float())   # 前向传播：通过像模型输入x计算预测的y
    y_pred = y_pred.float()
    for j in range(149):
        for i in range(22):
            if i % 2 == 0:
                y_pred[j, i] /= Z[j, 0]
            else:
                y_pred[j, i] /= Z[j, 1]
    y = y.float()
    loss = Loss(y_pred, y)  # 计算最终的训练误差
    print("训练完成, MSEloss为 {}".format(loss.item()))
    return net


###################模型测试#########
def test(x, y, net_path, Loss, Z):
    #net = torch.load(net_path)
    y_pred = net(x.float())
    y_pred = y_pred.float()
    y = y.float()
    #print('y_pred.shape',y_pred.shape)
    for j in range(51):
        for i in range(22):
            if i % 2 == 0:
                y_pred[j, i] /= Z[j, 0]
            else:
                y_pred[j, i] /= Z[j, 1]
    y_pred = y_pred.int()
    loss = (Loss(y_pred, y))  # 计算loss
    #print(y_pred[0, :])
    #print(y[0, :])
    #print(y_pred[0,:])
    print("测试完成, MSEloss为 {}".format(loss.item()))
    #print("测试正确率为 {}".format(aa/(aa+bb)))
    #print("测试错误率为 {}".format(bb/(aa+bb)))
    return 0


if __name__ == '__main__':
    start = time.clock()
    path = 'C:/Users/86181/Desktop/人工智能实验期中大作业/脊柱疾病智能诊断/'  #在不同机器上运行时，将path改成相应地址即可
    split = 'train/data'
    num = 149    #训练集样本数
    X_train, Y_train, Z_train = get_data(path, split, num)  #获取训练集数据
    split = 'test/data'
    num = 51     #测试集样本数
    X_test, Y_test, Z_test = get_data(path, split, num)    #获取测试集数据
    #定义网络、误差和优化器
    net = define_model()
    Loss = define_loss()
    optimizer = define_optimizer(1e-2)
    #进行训练
    Net1 = train(X_train, Y_train, net, Loss, optimizer,Z_train,path)
    #进行测试
    test(X_test , Y_test, Net1, Loss, Z_test)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))