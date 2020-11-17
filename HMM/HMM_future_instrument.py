#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
import quandl
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
# from sklearn.externals import joblib
import pandas as pd
import joblib
from backtest_func import yearsharpRatio, maxRetrace, annROR
import copy
import warnings

warnings.filterwarnings("ignore")


# 获取集合的所有子集
def PowerSetsRecursive(items):
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.volume = pn_data['volume']
        # self.amount=pn_data['amount']
        # self.returns = self.close-self.close.shift(1)

    def alpha000(self):
        data_m = self.close.rolling(10).apply(values_deviation, raw=True)
        return data_m


# 特征工程和建模
def get_best_hmm_model(X, max_states, max_iter=10000):
    best_score = -(10 ** 10)
    best_state = 0

    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components=state, random_state=100, covariance_type='diag',
                                n_iter=max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state
    best_model = GaussianHMM(n_components=best_state, random_state=100, covariance_type='diag', n_iter=max_iter).fit(X)
    return best_model


# Normalizde
def std_normallized(vals):
    return np.std(vals) / np.mean(vals)


# Ratio of diff
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]


# z-score
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


class DataFactory(object):
    def __init__(self, instrument, factor_lst, future_period, s_date, e_date):
        self.instrument = instrument
        self.s_date = s_date
        self.e_date = e_date
        self.factor_lst = factor_lst
        self.future_period = future_period

    # 获取行情数据
    def get_history(self):
        data_ori = pd.read_csv('C:/e/data/future_index/' + self.instrument + '_daily_index.csv')
        data_ori['date_time'] = data_ori['date_time'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
        data_ori.index = data_ori['date_time']
        data_ori['close_1'] = data_ori['close'].shift(1)
        data_ori = data_ori.loc[self.s_date:self.e_date, :]
        return data_ori

    # 获取训练集初始数据
    def get_dataset(self):
        data_ori = self.get_history()
        dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume']].shift(1)
        for alpha in self.factor_lst:
            alpha = 'Alpha.' + alpha
            Alpha = Alphas(dataset)
            dataset[alpha[-8:]] = eval(alpha)()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = pd.concat(
            [dataset[self.factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'date_time', 'close_1']]],
            axis=1)
        # 日涨跌幅，模型训练的Y值
        dataset['ret'] = dataset['open'].shift(-self.future_period) / dataset['open'] - 1
        dataset['ret'] = dataset['ret'].fillna(0)
        dataset = dataset.dropna()
        return dataset


class HmmStrategy(object):
    def __init__(self, asset, s_date, max_states, leverage, fee, data_set, cols_features, train_days=90, test_days=60,
                 insample=False, **kwargs):
        self.asset = asset
        self.s_date = s_date
        self.test_days = test_days
        self.train_days = train_days
        self.max_states = max_states
        self.leverage = leverage
        self.fee = fee
        self.data_set = data_set
        self.cols_features = cols_features
        self.insample = insample
        self.ret = pd.DataFrame()
        self.signal_state_all = pd.DataFrame()
        self.annR = 0
        self.sharp = 0
        self.max_retrace = 0

    # 隐状态定义涨跌：单一状态下累计损益为正即为看涨信号，反之为看跌信号
    def get_longshort_state_from_cumsum(self, dataset):
        dataset['date_time'] = pd.to_datetime(dataset['date_time'])
        net_new = copy.deepcopy(dataset).dropna()
        X = np.column_stack([net_new.loc[:, self.cols_features]])
        model = self.get_best_hmm_model(X)
        hidden_states = model.predict(X)
        long_states = []
        short_states = []
        random_states = []
        for k in range(model.n_components):
            idx = (hidden_states == k)
            idx_int = idx.astype(int)
            net_new['%dth_hidden_state' % k] = idx_int
            net_new['%dth_ret' % k] = net_new['%dth_hidden_state' % k] * net_new['ret']
            if net_new['%dth_ret' % k].sum() > 0:
                long_states.append(k)
            elif net_new['%dth_ret' % k].sum() < 0:
                short_states.append(k)
            elif net_new['%dth_ret' % k].sum() == 0:
                random_states.append(k)
        #         print('做多隐状态：%s, 做空隐状态：%s, 空仓隐状态：%s' %(long_states, short_states, random_states))
        return model, hidden_states, long_states, short_states, random_states

    def get_best_hmm_model(self, X, max_iter=10000):
        best_score = -(10 ** 10)
        best_state = 0
        for state in range(1, self.max_states + 1):
            hmm_model = GaussianHMM(n_components=state, random_state=100, covariance_type='diag',
                                    n_iter=max_iter).fit(X)
            try:
                if hmm_model.score(X) > best_score:
                    best_score = hmm_model.score(X)
                    best_state = state
            except:
                continue
        best_model = GaussianHMM(n_components=best_state, random_state=100, covariance_type='diag',
                                 n_iter=max_iter).fit(X)
        return best_model

    def handle_data(self, test_set, model, hidden_states, long_states, short_states, random_states):
        test_set['state'] = hidden_states[-len(test_set):]
        fee = self.fee
        state = None
        signal_lst = []
        trad_times = 0
        chg = 0
        chg_lst = []
        pos_lst = []
        pos = 0
        low_price_pre = 0
        high_price_pre = 100000000
        long_stop = False
        short_stop = False
        state_lst = []
        for idx, _row in test_set.iterrows():
            test_data = test_set[self.cols_features].loc[:idx, :]
            state = _row.state
            state_lst.append(state)
            if pos == 0:
                if state in long_states:
                    pos = 1
                    cost = _row.open * (1 + fee)
                    s_time = _row.date_time
                    hold_price = []
                    high_price = []
                    hold_price.append(cost / (1 + fee))
                    high_price.append(cost / (1 + fee))
                    high_price.append(_row.high)
                    chg = (pos * _row.close / cost + (1 - pos)) - 1
                elif state in short_states:
                    cost = _row.open * (1 - fee)
                    pos = -1
                    s_time = _row.date_time
                    hold_price = []
                    low_price = []
                    hold_price.append(cost / (1 - fee))
                    low_price.append(cost / (1 - fee))
                    low_price.append(_row.low)
                    chg = ((1 + pos) - pos * (2 - _row.close / cost)) - 1
                else:
                    chg = 0
            elif pos > 0:
                if state in short_states:
                    s_price = _row.open * (1 - fee)
                    trad_times += 1
                    net1 = pos * s_price / _row.close_1 + (1 - pos)
                    ret = s_price / cost - 1
                    e_time = _row.date_time
                    signal_row = []
                    signal_row.append(s_time)
                    signal_row.append(e_time)
                    signal_row.append(cost)
                    signal_row.append(s_price)
                    signal_row.append(ret)
                    signal_row.append((max(high_price) / cost) - 1)
                    signal_row.append(len(hold_price))
                    signal_row.append(pos)
                    signal_row.append('spk')
                    signal_lst.append(signal_row)
                    s_time = _row.date_time
                    cost = s_price
                    pos = -1
                    net2 = (1 + pos) - pos * (2 - _row.close / cost)
                    hold_price = []
                    low_price = []
                    hold_price.append(cost / (1 - fee))
                    low_price.append(cost / (1 - fee))
                    low_price.append(_row.low)
                    chg = net1 * net2 - 1
                    short_stop = False
                    long_stop = False
                elif state in random_states:
                    s_price = _row.open * (1 - fee)
                    trad_times += 1
                    net1 = pos * s_price / _row.close_1 + (1 - pos)
                    ret = s_price / cost - 1
                    e_time = _row.date_time
                    signal_row = []
                    signal_row.append(s_time)
                    signal_row.append(e_time)
                    signal_row.append(cost)
                    signal_row.append(s_price)
                    signal_row.append(ret)
                    signal_row.append((max(high_price) / cost) - 1)
                    signal_row.append(len(hold_price))
                    signal_row.append(pos)
                    signal_row.append('sp')
                    signal_lst.append(signal_row)
                    pos = 0
                    high_price.append(_row.high)
                    long_stop = False
                    short_stop = False
                    chg = net1 - 1
                else:
                    high_price.append(_row.high)
                    hold_price.append(_row.close)
                    chg = (pos * _row.close / _row.close_1 + (1 - pos)) - 1
            elif pos < 0:
                if state in long_states:
                    b_price = _row.open * (1 + fee)
                    e_time = _row.date_time
                    trad_times += 1
                    net1 = (1 + pos) - pos * (2 - b_price / _row.close_1)
                    ret = (cost - b_price) / cost
                    signal_row = []
                    signal_row.append(s_time)
                    signal_row.append(e_time)
                    signal_row.append(cost)
                    signal_row.append(b_price)
                    signal_row.append(ret)
                    signal_row.append((cost - min(low_price)) / cost)
                    signal_row.append(len(hold_price))
                    signal_row.append(pos)
                    signal_row.append('bpk')
                    signal_lst.append(signal_row)
                    pos = 1
                    cost = b_price
                    net2 = pos * _row.close / cost + 1 - pos
                    s_time = _row.date_time
                    hold_price = []
                    high_price = []
                    hold_price.append(cost / (1 + fee))
                    high_price.append(cost / (1 + fee))
                    high_price.append(_row.high)
                    chg = net1 * net2 - 1
                    short_stop = False
                    long_stop = False
                elif state in random_states:
                    b_price = _row.open * (1 + fee)
                    trad_times += 1
                    net1 = (1 + pos) - pos * (2 - b_price / _row.close_1)
                    ret = (cost - b_price) / cost
                    e_time = _row.date_time
                    signal_row = []
                    signal_row.append(s_time)
                    signal_row.append(e_time)
                    signal_row.append(cost)
                    signal_row.append(b_price)
                    signal_row.append(ret)
                    signal_row.append((cost - min(low_price)) / cost)
                    signal_row.append(len(hold_price))
                    signal_row.append(pos)
                    signal_row.append('bp')
                    signal_lst.append(signal_row)
                    pos = 0
                    low_price.append(_row.low)
                    long_stop = False
                    short_stop = False
                    chg = net1 - 1

                else:
                    low_price.append(_row.low)
                    hold_price.append(_row.close)
                    chg = ((1 + pos) - pos * (2 - _row.close / _row.close_1)) - 1
            chg_lst.append(chg)
            pos_lst.append(pos)
        ret = test_set.loc[:, ['close', 'date_time']]
        ret['chg'] = chg_lst
        ret['pos'] = pos_lst
        ret['state'] = state_lst
        signal_state_all = pd.DataFrame(signal_lst, columns=[
            's_time', 'e_time', 'b_price', 's_price', 'ret', 'max_ret', 'hold_day',
            'position', 'bspk'])
        return ret, signal_state_all

    def _analyze(self):
        # Summary output
        ret = self.ret
        ret = ret[ret.index >= self.s_date]
        ret.index = pd.to_datetime(ret.index)
        ret['net_close'] = ret['close'] / ret['close'].tolist()[0]
        ret['net'] = ret['net'] / ret['net'].tolist()[0]
        net_lst = ret['net'].tolist()
        annR = annROR(net_lst, 1)
        sharp = yearsharpRatio(net_lst, 1)
        max_retrace = maxRetrace(net_lst, 1)
        self.annR = annR
        self.sharp = sharp
        self.max_retrace = max_retrace

        f = plt.figure(figsize=(15, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']

        # Plot return
        ax1 = f.add_subplot(111)
        ax1.plot(ret.net, 'blue', label='net_hmm')
        ax1.plot(ret.net_close, 'red', label='net_close')
        ax1.legend()
        ax1.set_title('%s特征因子%s训练窗口%s测试窗口%s净值曲线' % (self.asset, self.cols_features, self.train_days, self.test_days))
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')

    def _get_ret_insample(self):
        self.insample = True
        ret_df = []
        signal_state_all = []
        train_days = self.train_days
        for i in range(0, len(self.data_set), train_days):
            if i + train_days > len(self.data_set):
                train_set = self.data_set.iloc[-train_days:]
                test_set = self.data_set.iloc[i:]
            else:
                train_set = self.data_set.iloc[i:i + train_days]
                test_set = train_set
            model, hidden_states, long_states, short_states, random_states = self.get_longshort_state_from_cumsum(
                train_set)
            ret, signal_state = self.handle_data(test_set, model, hidden_states, long_states, short_states,
                                                 random_states)
            ret_df.append(ret)
            signal_state_all.append(ret)
        ret_df = pd.concat(ret_df)
        ret_df['net'] = (1 + ret_df['chg']).cumprod()
        signal_state_all = pd.concat(signal_state_all)
        self.ret = ret_df
        self.signal_state_all = signal_state_all

    def _get_ret_outsample(self):
        self.insample = False
        ret_df = []
        signal_state_all = []
        train_days = self.train_days
        test_days = self.test_days
        for i in range(train_days, len(self.data_set), test_days):
            if i + test_days > len(self.data_set):
                all_set = self.data_set.iloc[i - train_days:i + test_days]
                train_set = self.data_set.iloc[i - train_days:i]
                test_set = self.data_set.iloc[i:]
            else:
                all_set = self.data_set.iloc[i - train_days:i + test_days]
                train_set = self.data_set.iloc[i - train_days:i]
                test_set = self.data_set.iloc[i:i + test_days]
            model, hidden_states, long_states, short_states, random_states = self.get_longshort_state_from_cumsum(
                train_set)
            #             print(len(hidden_states))
            hidden_states_predict = model.predict(test_set[self.cols_features])
            #             print(len(hidden_states_predict))
            ret, signal_state = self.handle_data(test_set, model, hidden_states_predict, long_states, short_states,
                                                 random_states)
            ret_df.append(ret)
            signal_state_all.append(ret)
        ret_df = pd.concat(ret_df)
        ret_df['net'] = (1 + ret_df['chg']).cumprod()
        signal_state_all = pd.concat(signal_state_all)
        self.ret = ret_df
        self.signal_state_all = signal_state_all

    def run_insample(self):
        self._get_ret_insample()
        self._analyze()
        return self.annR, self.sharp, self.max_retrace

    def run_outsample(self):
        self._get_ret_outsample()
        self._analyze()
        return self.annR, self.sharp, self.max_retrace


# 第二个关键 是通过特征来研究每个状态。
# 在此之后，我们可以将这两个事件（未来走势和当前状态）联系起来。
# 让我们为每个状态的特征编写代码和可视化。
def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m, m + h


def compare_hidden_states(hmm_model, cols_features, conf_interval, iters=1000):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    for i in range(0, hmm_model.n_components):
        mc_df = pd.DataFrame()

        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features
        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color=colours[i])
            axs[k][i].set_title(cols_features[k] + ' (state ' + str(i) + '): ' +
                                str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)
    plt.tight_layout()


def plot_hidden_states(model, data, X, column_price):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(model.n_components, 3, figsize=(15, 15))
    colours = cm.prism(np.linspace(0, 1, model.n_components))
    hidden_states = model.predict(X)
    print(hidden_states)
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        #         print(mask)
        #         print(data['future_return'][mask])
        ax[0].plot(data.index, data[column_price], c='grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c=colour)
        ax[0].set_title('{0}th hidder state'.format(i))
        ax[0].grid(True)

        ax[1].hist(data['future_return'][mask], bins=30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title('future return distrbution at {0}th hidder state'.format(i))
        ax[1].grid(True)

        ax[2].plot(data['future_return'][mask].cumsum(), c=colour)
        ax[2].set_title('cummulative future return at {0}th hidden state'.format(i))
        ax[2].grid(True)
    plt.tight_layout()


def analyse_hidden_state_of_single_factor(dataset, factor_list, max_states):
    dataset['date_time'] = pd.to_datetime(dataset['date_time'])
    for i in factor_list:
        net_new = copy.deepcopy(dataset)
        net_new[i] = dataset[i]
        X = np.column_stack([net_new[i]])
        model = get_best_hmm_model(X, max_states)
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
            plt.plot_date(net_new['date_time'][idx], net_new['close'][idx], '.', label='%dth hidden state' % k,
                          lw=1)

            # plt.legend()
            # plt.grid(1)
        plt.tight_layout()
        #         plt.savefig('c:/e/' + 'fig/%s.png' % i)
        plt.figure(figsize=(15, 8))
        colnume_name = ['%dth_net' % k for k in range(model.n_components)]
        colnume_name.append('date_time')
        plt.plot(net_new[colnume_name].set_index(['date_time']))
        #         net_new[colnume_name].set_index(['date_time']).plot()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title('特征%s各个隐状态多头累计收益' % i)
        plt.tight_layout()



if __name__ == '__main__':
    # ## 获取行情数据

    data_path = 'C:/e/data/future_index/'
    future_period = 1
    asset = 'IF'
    max_states = 6
    factor_lst = ['alpha000']
    cols_features = factor_lst
    s_data_date = '2010-01-01'
    e_data_date = '2020-11-01'
    column_price = 'close'
    column_high = 'high'
    column_low = 'low'
    column_volume = 'volume'
    dataFactory = DataFactory(asset, factor_lst, future_period, s_data_date, e_data_date)
    data_ori = dataFactory.get_history()

    s_date = '2015-01-01'
    train_s_date = '2014-01-01'

    all_feature_lst = PowerSetsRecursive(factor_lst)
    all_feature_lst = [i for i in all_feature_lst if len(i) >= 1]
    # all_feature_lst = [['price_deviation']]
    train_days_lst = [240]
    test_days_lst = [30]
    period_param_state = []
    asset_lst = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                 'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                 'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                 'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']
    for feature_lst in all_feature_lst:
        for train_days in train_days_lst:
            for test_days in test_days_lst:
                for asset in asset_lst:
                    try:
                        dataFactory = DataFactory(asset, factor_lst, future_period, s_data_date, e_data_date)
                        dataset = dataFactory.get_dataset()
                        data_set = dataset[dataset.index >= train_s_date]
                        print(asset, '特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days)
                        strategy_obj = HmmStrategy(asset=asset, s_date='2015-01-01', max_states=max_states, leverage=1,
                                                   fee=0.001, data_set=data_set,
                                                   cols_features=feature_lst, train_days=train_days,
                                                   test_days=test_days, insample=False)
                        annR, sharp, max_retrace = strategy_obj.run_outsample()
                        print('年化收益: ' + str(annR))
                        print('夏普比率: ' + str(sharp))
                        print('最大回撤: ' + str(max_retrace))
                        period_param_state.append([asset, feature_lst, train_days, test_days, annR, sharp, max_retrace])
                    except:
                        continue
    period_param_state_df = pd.DataFrame(
        period_param_state, columns=['instrument', '特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])

    # In[70]:

    period_param_state_df.to_csv('c:/e/resualt.csv', encoding='gbk')

    dataset = dataFactory.get_dataset()
    all_feature_lst = PowerSetsRecursive(factor_lst)
    all_feature_lst = [i for i in all_feature_lst if len(i) > 0]
    # ## 特征因子样本内回测

    # ### 特征因子样本内不同窗口期回测

    s_date = '2015-01-01'
    train_s_date = '2014-01-01'
    data_set = dataset[dataset.index >= train_s_date].dropna()
    train_days = len(data_set)
    test_days = len(data_set)
    # all_feature_lst = [['price_deviation']]
    for feature_lst in all_feature_lst:
        for train_days in range(60, 301, 30):
            print('特征因子：', feature_lst, '训练窗口:', train_days)
            test_days = train_days
            strategy_obj = HmmStrategy(s_date=s_date, max_states=max_states, leverage=1, fee=0.0002, data_set=data_set,
                                       cols_features=feature_lst, train_days=train_days, test_days=test_days,
                                       insample=True)

            annR, sharp, max_retrace = strategy_obj.run_insample()
            print('年化收益: ' + str(annR))
            print('夏普比率: ' + str(sharp))
            print('最大回撤: ' + str(max_retrace))

    # ### 样本内不同特征因子回测

    # In[43]:

    all_feature_lst = PowerSetsRecursive(factor_lst)
    all_feature_lst = [i for i in all_feature_lst if len(i) > 0]
    s_date = '2015-01-01'
    data_set = dataset[dataset.index >= '2015-01-01'].dropna()
    train_days = 90
    test_days = train_days
    for feature_lst in all_feature_lst:
        print('特征因子：', feature_lst, '训练窗口:', train_days)
        strategy_obj = HmmStrategy(s_date=s_date, max_states=max_states, leverage=1, fee=0.0002, data_set=data_set,
                                   cols_features=feature_lst, train_days=train_days, test_days=test_days, insample=True)
        annR, sharp, max_retrace = strategy_obj.run_insample()
        print('年化收益: ' + str(annR))
        print('夏普比率: ' + str(sharp))
        print('最大回撤: ' + str(max_retrace))

    # ## 特征因子样本外回测

    # ### 样本外不同滚动窗口回测，参数敏感性测试

    # In[23]:

    data_set = dataset[dataset.index >= '2014-01-01']
    s_date = '2015-01-01'
    train_s_date = '2015-01-01'
    all_feature_lst = [['alpha000']]
    train_days_lst = [i for i in range(480, 50, -60)]
    test_days_lst = [i for i in range(60, 500, 60)]
    period_param_state = []
    for feature_lst in all_feature_lst:
        sharp_lst = []
        harvest = []
        for train_days in train_days_lst:
            harvest_row = []
            for test_days in test_days_lst:
                print('特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days)
                strategy_obj = HmmStrategy(s_date='2015-01-01', max_states=max_states, leverage=1, fee=0.0002,
                                           data_set=data_set,
                                           cols_features=feature_lst, train_days=train_days, test_days=test_days,
                                           insample=False)
                annR, sharp, max_retrace = strategy_obj.run_outsample()
                #             print('年化收益: ' + str(annR))
                #             print('夏普比率: ' + str(sharp))
                #             print('最大回撤: ' + str(max_retrace))
                sharp_lst.append(sharp)
                period_param_state.append([feature_lst, train_days, test_days, annR, sharp, max_retrace])
                harvest_row.append(sharp)
            harvest.append(harvest_row)
        x_label = train_days_lst
        y_label = test_days_lst
        # print(harvest)
        harvest = np.array(harvest)
        fig, ax1 = plt.subplots(figsize=(2 * len(y_label), len(y_label)), nrows=1)

        vmax = max(max(harvest[i]) for i in range(len(harvest)))
        vmin = -vmax
        #     print(vmax)
        if vmax < 0:
            vmin = min(min(harvest[i]) for i in range(len(harvest)))
        h = sns.heatmap(harvest, annot=True, fmt='.2f', ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'size': 20},
                        cbar=False)
        cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
        cb.ax.tick_params(labelsize=28)
        ax1.set_title('特征因子%s不同窗口期热力图' % (feature_lst), fontsize=32)
        ax1.set_xticklabels(y_label, fontsize=20)
        ax1.set_yticklabels(x_label, fontsize=20)
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }
        ax1.set_xlabel('test_days', fontsize=24)
        ax1.set_ylabel('train_days', fontsize=24)
        fig.tight_layout()
    period_param_state_df = pd.DataFrame(
        period_param_state, columns=['特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])

    # In[24]:

    period_param_state_df

    # In[29]:

    data_set = dataset[dataset.index >= '2014-01-01']
    s_date = '2015-01-01'
    train_s_date = '2015-01-01'
    all_feature_lst = [['price_deviation']]
    train_days_lst = [i for i in range(300, 190, -10)]
    test_days_lst = [2, 5, 10, 15, 20, 25, 30, 35, 40, 60]
    # train_days_lst = [i for i in range(480, 60, -60)]
    # test_days_lst = [i for i in range(1, len(train_days_lst))]
    period_param_state = []
    for feature_lst in all_feature_lst:
        sharp_lst = []
        harvest = []
        for train_days in train_days_lst:
            harvest_row = []
            for test_days in test_days_lst:
                print('特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days)
                strategy_obj = HmmStrategy(s_date='2015-01-01', max_states=max_states, leverage=1, fee=0.0002,
                                           data_set=data_set,
                                           cols_features=feature_lst, train_days=train_days, test_days=test_days,
                                           insample=False)
                annR, sharp, max_retrace = strategy_obj.run_outsample()
                #             print('年化收益: ' + str(annR))
                #             print('夏普比率: ' + str(sharp))
                #             print('最大回撤: ' + str(max_retrace))
                sharp_lst.append(sharp)
                period_param_state.append([feature_lst, train_days, test_days, annR, sharp, max_retrace])
                harvest_row.append(sharp)
            harvest.append(harvest_row)
        x_label = train_days_lst
        y_label = test_days_lst
        # print(harvest)
        harvest = np.array(harvest)
        fig, ax1 = plt.subplots(figsize=(2 * len(y_label), len(y_label)), nrows=1)

        vmax = max(max(harvest[i]) for i in range(len(harvest)))
        vmin = -vmax
        #     print(vmax)
        if vmax < 0:
            vmin = min(min(harvest[i]) for i in range(len(harvest)))
        h = sns.heatmap(harvest, annot=True, fmt='.2f', ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'size': 20},
                        cbar=False)
        cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
        cb.ax.tick_params(labelsize=28)
        ax1.set_title('特征因子%s不同窗口期热力图' % (feature_lst), fontsize=32)
        ax1.set_xticklabels(y_label, fontsize=20)
        ax1.set_yticklabels(x_label, fontsize=20)
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }
        ax1.set_xlabel('test_days', fontsize=24)
        ax1.set_ylabel('train_days', fontsize=24)
        fig.tight_layout()
    period_param_state_df = pd.DataFrame(
        period_param_state, columns=['特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])

    s_date = '2015-01-01'
    train_s_date = '2014-01-01'
    data_set = dataset[dataset.index >= train_s_date]
    all_feature_lst = PowerSetsRecursive(cols_features)
    all_feature_lst = [i for i in all_feature_lst if len(i) >= 1]
    # all_feature_lst = [['price_deviation']]
    train_days_lst = [240]
    test_days_lst = [60]
    period_param_state = []
    for feature_lst in all_feature_lst:
        for train_days in train_days_lst:
            for test_days in test_days_lst:
                print('特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days)
                strategy_obj = HmmStrategy(s_date='2015-01-01', max_states=max_states, leverage=1, fee=0.0002,
                                           data_set=data_set,
                                           cols_features=feature_lst, train_days=train_days, test_days=test_days,
                                           insample=False)
                annR, sharp, max_retrace = strategy_obj.run_outsample()
                print('年化收益: ' + str(annR))
                print('夏普比率: ' + str(sharp))
                print('最大回撤: ' + str(max_retrace))
                period_param_state.append([feature_lst, train_days, test_days, annR, sharp, max_retrace])
    period_param_state_df = pd.DataFrame(
        period_param_state, columns=['特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])

    # In[62]:

    pd.options.display.max_columns = 300
    pd.options.display.max_rows = 300

    # In[64]:

    period_param_state_df.sort_values(['夏普比率'], ascending=False)

    # In[ ]:

    # In[354]:

    dataset = dataset.dropna()

    # In[355]:

    dataset

    # In[356]:

    # Split the data on sets
    train_edate = '2020-11-01 00:00:00'
    train_ind = int(np.where(dataset.index == train_edate)[0])
    train_set = dataset[cols_features].values[:train_ind]
    test_set = dataset[cols_features].values[train_ind:]
    test_df = dataset[train_ind:]

    # In[357]:

    train_ind

    # In[358]:

    # Plot features 五个新的时间序列
    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(len(cols_features), 1, figsize=(15, 15))
    colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
    for i in range(0, len(cols_features)):
        axs[i].plot(dataset.reset_index()[cols_features[i]], color=colours[i])
        axs[i].set_title(cols_features[i])
        axs[i].grid(True)

    plt.tight_layout()

    # In[359]:

    # General plots of hidden states
