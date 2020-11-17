# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:26:21 2020

@author: zhangfang
"""
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import copy

sns.set_style('white')
'''
from WindPy import w 
w.stop()
w.start()
data_test=w.wsd("RB.SHF", "open,high,low,close,volume,pct_chg", "2000-02-14", "2016-03-01", "")
'''


# 定义上穿函数
def CrossOver(price1, price2):
    Con = [False]
    Precon = False
    conter = 0
    for i in range(1, len(price1)):
        if (price1[i] > price2[i]):
            conter = 1
            Con1 = price1[i - 1] == price2[i - 1]
            while (Con1 == True and i - conter > 0):
                conter = conter + 1
                Con1 = (price1[i - conter] == price2[i - conter])
            Precon = (price1[i - conter] < price2[i - conter])
            Con.append(Precon)
            conter = 0
        else:
            Con.append(False)
            conter = 0
    return Con


# 定义下穿函数
def CrossUnder(price1, price2):
    Con = [False]
    Precon = False
    conter = 0
    for i in range(1, len(price1)):
        if (price1[i] < price2[i]):
            conter = 1
            Con1 = price1[i - 1] == price2[i - 1]
            while (Con1 == True and i - conter > 0):
                conter = conter + 1
                Con1 = (price1[i - conter] == price2[i - conter])
            Precon = (price1[i - conter] > price2[i - conter])
            Con.append(Precon)
            conter = 0
        else:
            Con.append(False)
            conter = 0
    return Con


# 定义MACD函数
def EMA_MACO(data, d):
    test = pd.Series(index=range(len(data)))
    test = data.ewm(span=d).mean()
    return test


def MACD(data, FastLength, SlowLength, MACDLength):
    data['Diff'] = ''
    data['Diff'] = EMA_MACO(data['open'], FastLength) - EMA_MACO(data['open'], SlowLength)
    data['DEA'] = ''
    data['DEA'] = EMA_MACO(data['Diff'], MACDLength)
    data['MACD'] = ''
    data['MACD'] = data['Diff'] - data['DEA']
    return data


if __name__ == '__main__':
    data_path = 'c:/e/data/future_index/'
    resualt_path = 'c:/e/hmm/'
    # plt.style.use('ggplot')

    # 导入数据，生成因子
    data = pd.read_csv(data_path + 'RB_daily_index.csv')
    print(data)
    data = data.assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
    print(data)
    data['log_return'] = np.log(data['open'] / data['open'].shift(1))
    data['log_return'] = data['log_return'].fillna(0)
    data['log_return_5'] = np.log(data['open'] / data['open'].shift(5))
    data['log_return_5'] = data['log_return_5'].fillna(0)
    for h, k in [(5, 10), (5, 15), (5, 20), (10, 15), (10, 20), (15, 20), (15, 30)]:
        data['fast_line'] = ''
        data['slow_line'] = ''
        data['fast_line'] = data['open'].rolling(h).mean()
        data['slow_line'] = data['open'].rolling(k).mean()
        data['fast_line'] = data['fast_line'].fillna(value=data['open'].rolling(window=len(data['open']), min_periods=1).mean())
        data['slow_line'] = data['slow_line'].fillna(value=data['open'].rolling(window=len(data['open']), min_periods=1).mean())
        data['dist_%s_%s' % (k, h)] = data['fast_line'] - data['slow_line']
    for i in range(5, 31, 5):
        data['MA_%s' % i] = data['open'].rolling(i).mean()
        data['MA_%s' % i] = data['MA_%s' % i].fillna(0) - data['open']
    data = MACD(data, 12, 26, 9)
    for h in range(10, 26, 5):
        data['fast_line'] = ''
        data['slow_line'] = ''
        data['fast_line'] = data['high'].shift(1).rolling(h).mean()
        data['slow_line'] = data['low'].shift(1).rolling(h).mean()
        data['fast_line'] = data['fast_line'].fillna(value=data['high'].rolling(window=len(data['high']), min_periods=1).max())
        data['slow_line'] = data['slow_line'].fillna(value=data['low'].rolling(window=len(data['low']), min_periods=1).min())
        data['dist_high_%s' % h] = data['high'] - data['fast_line']
        data['dist_low_%s' % h] = data['low'] - data['slow_line']
    # 引入隐马尔科夫模型
    factor_list = ['close', 'volume', 'dist_10_5', 'dist_15_5', 'dist_20_5', 'dist_15_10', 'dist_20_10', 'dist_20_15',
                   'dist_30_15', 'log_return', 'log_return_5', 'MACD', 'dist_high_10', 'dist_high_15', 'dist_high_20',
                   'dist_high_25', 'dist_low_10', 'dist_low_15', 'dist_low_20', 'dist_low_25', 'MA_5', 'MA_10', 'MA_15',
                   'MA_20', 'MA_25', 'MA_30']
    # factor_list = ['dist_low_20']
    data['date_time'] = pd.to_datetime(data['date_time'])
    net = data[['date_time', 'close', 'open', 'high', 'low']]
    # net.loc[:, ['close', 'open', 'high', 'close']] = net.loc[:, ['close', 'open', 'high', 'close']].astype(float)
    print(type(net.close), type(net.open))
    net['ret'] = net['close'].shift(-1) / net['open'].shift(-1) - 1
    net['ret'] = net['ret'].fillna(0)
    print(net)

    for i in factor_list:
        net_new = copy.deepcopy(net)
        net_new[i] = data[i]
        X = np.column_stack([net_new[i]])
        model = GaussianHMM(n_components=6, covariance_type="diag", n_iter=1000, random_state=0).fit(X)
        hidden_states = model.predict(X)
        plt.figure(figsize=(15, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        for k in range(model.n_components):
            idx = (hidden_states == k)
            idx_int = idx.astype(int)
            net_new['%dth_hidden_state' % k] = idx_int
            net_new['%dth_ret' % k] = net_new['%dth_hidden_state' % k] * net_new['ret']
            net_new['%dth_net' % k] = net_new['%dth_ret' % k].cumsum()
            # print(idx)
            plt.title('特征%s标记在收盘价序列上的隐状态' % i)
            plt.plot_date(net_new['date_time'][idx], net_new['close'][idx], '.', label='%dth hidden state' % k, lw=1)
            # plt.legend()
            # plt.grid(1)
        plt.savefig(resualt_path + 'fig/%s.png' % i)
        plt.figure(figsize=(15, 8))
        colnume_name = ['%dth_net' % k for k in range(model.n_components)]
        colnume_name.append('date_time')
        net_new[colnume_name].set_index(['date_time']).plot()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title('特征%s各个隐状态多头累计收益' % i)
        plt.savefig(resualt_path + 'fig/%s_cum_ret.png' % i)
