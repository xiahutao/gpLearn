# coding=utf-8
import time
import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
# 解决中文乱码方法
from pylab import mpl


def normalize(data):  # 均值标准差归一化函数
    mu = torch.mean(data)
    std = torch.std(data)
    return (data - mu) / std


def guiyimaxmin(x):  # 最大最小值归一化函数
    min = torch.min(x)
    max = torch.max(x)
    return (x - min) / (max - min)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Net(torch.nn.Module):  # 设计前项运算结果的网络
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 1, 2, 0),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, 2, 0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(384, 6)
        )

    def forward(self, x):  # Conv1d数据格式规定（900,1,4)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        print(x)
        return x


if __name__ == '__main__':
    data_path = 'c:/e/lstm/'
    mpl.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体FangSong仿宋
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    train_use = 0  # 0训练网络，找到规律后保存成文件，1直接读取保存的规律文件，来进行实际预测,2直接读取保存的规律文件，来进行实际预测
    filename = __file__.split(".")[0].split('/')[-1]

    trainxycsv = pd.read_csv(data_path + 'ocm_x_train_20.csv', index_col=0, header=0)  # 把csv文件里的内容读出来
    testxycsv = pd.read_csv(data_path + 'ocm_x_test_20.csv', index_col=0, header=0)  # 把csv文件里的内容读出来
    prexycsv = pd.read_csv(data_path + 'ocm_x_pre_20.csv', index_col=0, header=0)  # 把csv文件里的内容读出来
    train_data = torch.tensor(trainxycsv.values, dtype=torch.float)  # 把xy数据转化成pytorch的tensor格式
    test_data = torch.tensor(testxycsv.values, dtype=torch.float)  # 把xy数据转化成pytorch的tensor格式
    pre_data = torch.tensor(prexycsv.values, dtype=torch.float)  # 把xy数据转化成pytorch的tensor格式
    test_data_count = 500  # 最后留100个数据作为测试用
    xtrain = train_data[:, 0:-2]  # 把1000个数据中的900个作为训练的数据x
    # xtrain = train_data[:, 0:10]
    # xtrain = guiyimaxmin(xtrain)  # 把数据归一化一下
    ytrain = train_data[:, [-1]]  # 把1000个数据中的900个作为训练的数据y
    xtest = test_data[:, 0:-2]  # 把1000个数据中的最后100个作为测试数据x
    # xtest = test_data[:, 0:10]
    # xtest = guiyimaxmin(xtest)  # 把数据归一化一下
    ytest = test_data[:, [-1]]  # 把1000个数据中的最后100个作为测试数据

    xpredict = pre_data[:, 0:-2]  # 把1000个数据中的最后100个作为验证数据x
    # xpredict = pre_data[:, 0:10]
    # xtest = guiyimaxmin(xtest)  # 把数据归一化一下
    ypredict = pre_data[:, [-1]]  # 把1000个数据中的最后100个作为测试数据
    # ypre_value = data[-test_data_count:, [-2]]
    ypre_value = prexycsv['ret'].tolist()
    # print(ypre_value)
    # print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 定义一个运算设备，如果电脑有GPU就用GPU运算，如果没有就用CPU运算
    model = Net().to(device)  # 把网络模型赋值给model，并且把模型放到GPU上运行
    print(len(xtrain), len(xtrain[0]))
    testnetx = torch.randn((len(xtrain), len(xtrain[0]))).to(device)  # 把x的值放到GPU上执行
    testnety = model(testnetx)
    criteon = nn.CrossEntropyLoss().to(device)  # 定义损失函数类型，并且把它放到GPU上运行
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义优化方式，梯度下降算法参数调整
    if train_use == 0:
        trainlosslist = []  # 定义在训练时,累计每个epoch的loss值存储数组
        testlosslist = []  # 定义在测试时,累计每个epoch的loss值存储数组
        testacclist = []  # 定义在测试时,累计每个epoch的预测和真实的准确率值存储数组
        plt.ion()  # 开启plt画图，动态图模式
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        for epoch in range(700):  # 循环训练200次
            model.train()  # 切换到训练模式
            xtrain = xtrain.to(device)  # 把训练x放到GPU上执行
            ytrain = ytrain.to(device)  # 把训练y放到GPU上执行
            ytrain = ytrain.t().squeeze(0).long()
            logits = model(xtrain)  # 通过网络模型的运算，得到预测值
            loss = criteon(logits, ytrain)  # 通过损失函数的运算，得到损失值
            optimizer.zero_grad()  # 清空w，b的导数
            loss.backward()  # 每次网络的W,B全部自动求导计算出导数
            optimizer.step()  # 根据你定义的梯度下降规则来更新每层网络的W和b
            trainlosslist.append(loss.item())  # 把这一轮训练计算得出的loss值放入trainlosslist数组
            model.eval()  # 切换到测试模式
            with torch.no_grad():  # 测试模式，不需要任何w，b的导数值
                xtest = xtest.to(device)  # 把测试x放到GPU上执行
                ytest = ytest.to(device)  # 把测试y放到GPU上执行
                ytest = ytest.t().squeeze(0).long()
                logits = model(xtest)  # 通过网络模型的运算，得到预测值
                testloss = criteon(logits, ytest)  # 通过损失函数的运算，得到损失值
                testlosslist.append(testloss.item())  # 把这一轮测试计算得出的loss值放入testlosslist数组
                correct = torch.eq(logits.argmax(dim=1), ytest).float().sum().item()  # 计算所有预测正确的个数
                acc = (correct / xtest.size(0)) * 100  # 预测正确的个数除以总数，得出预测的正确百分比
                testacclist.append(acc)  # 把这一轮测试计算得出的acc值放入testacclist数组
                if epoch % 10 == 0:  # 每10轮画一次图
                    ax1.cla()  # 因为是动态图，所以先擦除上一张图
                    l3 = ax1.scatter(np.arange(ytest.shape[0]), ytest.to(device='cpu'))  # 把100个真实结果点画出来
                    l4 = ax1.scatter(np.arange(logits.argmax(dim=1).shape[0]),
                                     logits.argmax(dim=1).to(device='cpu'))  # 把100个预测结果点画出来
                    ax1.legend([l3, l4], ['%d个值的真实结果' % ytest.shape[0], '%d个值的预测结果' % ytest.shape[0]],
                               loc='best')  # 显示图例
                    ax1.text(0.5, 0, '训练次数=%d' % epoch, fontdict={'size': 20, 'color': 'red'})  # 显示不断更新的训练次数
            ax3.cla()  # 因为是动态图，所以先擦除上一张图
            ax3.plot(testacclist)  # 把每一轮的测试正确率acc值画出来
            ax3.text(0.5, 0, '测试集正确率')
            ax2.cla()  # 因为是动态图，所以先擦除上一张图
            l1, = ax2.plot(trainlosslist)  # 把每一轮的训练loss值画出来
            l2, = ax2.plot(testlosslist)  # 把每一轮的测试loss值画出来
            ax2.legend([l1, l2], ['tranloss', 'testloss'], loc='best')  # 显示图例
            plt.xlabel('epochs')  # 画的图x轴，标注epochs字样
            plt.pause(0.1)  # 暂停0.2秒，以免画的太快感觉不到图在动
            plt.ioff()  # 结束动态图模式
        plt.show()  # 最终显示图片
        torch.save(model, filename + '.pkl')
    if train_use != 1:
        model = torch.load(filename + '.pkl')
        y_true_value = ypredict
        y_true_value = y_true_value.t().squeeze(0).long()
        y_ret_value = ypre_value
        xpredict = xpredict.to(device)
        logits = model(xpredict)

        # y_ret_value = y_ret_value.t().squeeze(0).long()
        # print(y_ret_value)
        b_lst = []
        for i in range(len(xpredict)):
            x_ = xpredict[i]
            x_ = x_.to(device)
            x_ = x_.unsqueeze(0)
            logits_ = model(x_)
            b = logits_.argmax(dim=1)[0]
            b_lst.append(b)
            print(b)

        correct = torch.eq(logits.argmax(dim=1), y_true_value).float().sum().item()  # 计算所有预测正确的个数
        ret = pd.DataFrame({'predict': b_lst, 'true': y_true_value, 'ret': y_ret_value})
        print(len(ret))
        ret = ret[(ret['predict'] != 2) & (ret['predict'] != 3)]
        ret['predict'] = ret['predict'].apply(lambda x: 0 if x < 2 else 2)
        print(len(ret))

        ret['ret_y'] = ret['ret'].apply(lambda x: 0 if x > 0 else 2)
        ret['acc'] = ret['predict'] - ret['ret_y']
        ret.to_csv(data_path + 'pre.csv')

        acc = (correct / xpredict.size(0)) * 100  # 预测正确的个数除以总数，得出预测的正确百分比
        ave_profit = ret[ret['acc']==0]
        ave_profit['ret'] = abs(ave_profit['ret'])
        ave_profit = ave_profit['ret'].mean()
        ave_loss = ret[ret['acc']!=0]
        ave_loss['ret'] = abs(ave_loss['ret'])
        ave_loss = ave_loss['ret'].mean()
        odd = ave_profit/ave_loss

        acc_bs = len(ret[ret['acc']==0])/len(ret)
        acc_s = len(ret[(ret['acc'] == 0) & (ret['ret_y'] == 2)]) / len(ret[ret['predict'] == 2])
        acc_b = len(ret[(ret['acc'] == 0) & (ret['ret_y'] == 0)]) / len(ret[ret['predict'] == 0])
        print('acc：%s, acc_bs：%s, acc_s：%s, acc_b：%s, odd:%s' %(acc, acc_bs, acc_s, acc_b, odd))
    if train_use == 1:
        timeold = ''
        while (True):
            with open(data_path + 'truedata.txt', mode='r', encoding='utf_16_le') as f:
                rd = f.read()
                if len(rd) <= 0:
                    continue
                time1 = rd.split(';')[0]
                if timeold != time1:
                    data = rd.split(';')[2].split(',')
                    x = list(map(eval, data))
                    model_load = torch.load(filename + '.pkl')
                    x = torch.tensor(x, dtype=torch.float).to(device)  # 把x的值放到GPU上执行
                    x = x.unsqueeze(0)
                    y = model_load(x)  # 得到y值，但是y值里包含了3个输出
                    y = y.squeeze(0)  # 减少一个维度
                    y = y.argmax(dim=0)  # 获取3个输出中概率最大的那个值的所在序号，也就是最终的结果值
                    y = y.item()  # 把结果从pytorch的tenser类型转化为普通的float类型
                    print(y)  # 打印出最终预测结果
                    try:
                        f1 = open(data_path + 'jieguo.txt', mode='w', encoding='utf_16_le')
                        localtime = time.localtime(time.time())
                        print(time.strftime("%Y.%m.%d %H:%M:%S" + ',' + str(y), localtime))
                        f1.write(
                            str(math.floor(time.time())) + ',' + time.strftime("%Y.%m.%d %H:%M:%S",
                                                                               localtime) + ',' + str(
                                y))
                        timeold = time1
                    except IOError:
                        timeold = ''
                    finally:
                        f1.close()
