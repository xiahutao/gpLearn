# %%

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


# Normalizde
def std_normallized(vals):
    return np.std(vals) / np.mean(vals)


# Ratio of diff
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]


# z-score
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)


# 获取集合的所有子集
def PowerSetsRecursive(items):
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


def get_index(i, N):
    if i < N:
        return 0
    else:
        return i - N


class DataFactory(object):
    def __init__(self, instrument, factor_lst, future_period, s_date, e_date, data_ori):
        self.instrument = instrument
        self.s_date = s_date
        self.e_date = e_date
        self.factor_lst = factor_lst
        self.future_period = future_period
        self.data_ori = data_ori

    # 获取行情数据
    def get_history(self):
        data_ori = self.data_ori
        data_ori['date_time'] = data_ori['date_time'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
        data_ori.index = data_ori['date_time']
        data_ori['close_1'] = data_ori['close'].shift(1)
        data_ori = data_ori.loc[self.s_date:self.e_date, :]
        return data_ori

    # 获取训练集初始数据
    def get_dataset_close(self):
        data_ori = self.get_history()
        dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        for alpha in self.factor_lst:
            alpha = 'Alpha.' + alpha
            Alpha = Alphas(dataset)
            dataset[alpha[-8:]] = eval(alpha)()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = pd.concat(
            [dataset[self.factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'date_time', 'close_1']]],
            axis=1)
        # 日涨跌幅，模型训练的Y值
        dataset['ret'] = dataset['close'].shift(-self.future_period) / dataset['close'] - 1
        dataset['ret'] = dataset['ret'].fillna(0)
        dataset = dataset.dropna()
        return dataset

    def get_dataset_open(self):
        data_ori = self.get_history()
        dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        for alpha in self.factor_lst:
            alpha = 'Alpha.' + alpha
            Alpha = Alphas(dataset)
            dataset[alpha[-8:]] = eval(alpha)()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = pd.concat(
            [dataset[self.factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'date_time', 'close_1']]],
            axis=1)
        # 日涨跌幅，模型训练的Y值
        dataset['ret'] = dataset['open'].shift(-self.future_period-1) / dataset['open'].shift(-1) - 1
        dataset['ret'] = dataset['ret'].fillna(0)
        dataset = dataset.dropna()
        return dataset

    def get_dataset_aveprice(self):
        data_ori = self.get_history()
        data_ori['ave_price'] = data_ori['money'] / data_ori['volume']
        dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume', 'ave_price']]
        for alpha in self.factor_lst:
            alpha = 'Alpha.' + alpha
            Alpha = Alphas(dataset)
            dataset[alpha[-8:]] = eval(alpha)()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = pd.concat(
            [dataset[self.factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'date_time', 'close_1']]],
            axis=1)
        # 日涨跌幅，模型训练的Y值
        dataset['ret'] = dataset['ave_price'].shift(-self.future_period-1) / dataset['ave_price'].shift(-1) - 1
        dataset['ret'] = dataset['ret'].fillna(0)
        dataset = dataset.dropna()
        return dataset


class HmmStrategy(object):
    def __init__(self, asset, s_date, max_states, leverage, fee, data_set, cols_features, price_name, type, train_days=90, test_days=60,
                 insample=False, **kwargs):
        self.price_name = price_name
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
        self.type = type

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

    def handle_data_stock(self, test_set, hidden_states, long_states, short_states, random_states):
        test_set['state'] = hidden_states[-len(test_set):]
        fee = self.fee
        state = None
        signal_lst = []
        trad_times = 0
        chg = 0
        chg_lst = []
        pos_lst = []
        pos = 0.5
        price_name = self.price_name
        state_lst = []
        for idx, _row in test_set.iterrows():
            state = _row.state
            state_lst.append(state)
            if pos == 0.5:
                if state in long_states:
                    cost = _row[price_name] * (1 + fee)
                    pos = 1
                    hold_price = []
                    high_price = []
                    hold_price.append(cost / (1 + fee))
                    high_price.append(cost / (1 + fee))
                    high_price.append(_row.high)
                    chg = 0.5*_row.close / _row.close_1 + 0.5*_row.close / cost - 1
                elif state in short_states:
                    cost = _row[price_name] * (1 - fee)
                    pos = 0
                    s_time = _row.date_time
                    hold_price = []
                    low_price = []
                    hold_price.append(cost / (1 - fee))
                    low_price.append(cost / (1 - fee))
                    low_price.append(_row.low)
                    chg = 0.5*0 + 0.5 * (cost-_row.close_1) / _row.close_1
                else:
                    chg = 0.5*(_row.close-_row.close_1) / _row.close_1 + 0.5*0
            elif pos == 1:
                if state in short_states:
                    s_price = _row[price_name] * (1 - fee)
                    pos = 0
                    s_time = _row.date_time
                    chg = (s_price-_row.close_1) / _row.close_1
                elif state in random_states:
                    s_price = _row[price_name] * (1 - fee)
                    trad_times += 1
                    pos = 0.5
                    chg = 0.5 * (_row.close-_row.close_1) / _row.close_1 + 0.5 * (s_price-_row.close_1) / _row.close_1
                else:
                    chg = (_row.close-_row.close_1) / _row.close_1
            elif pos == 0:
                if state in long_states:
                    b_price = _row[price_name] * (1 + fee)
                    pos = 1
                    chg = (_row.close-b_price) / b_price
                elif state in random_states:
                    b_price = _row[price_name] * (1 + fee)
                    pos = 0.5
                    chg = (_row.close-b_price) / b_price * 0.5
                else:
                    chg = 0
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

    def handle_data(self, test_set, hidden_states, long_states, short_states, random_states):
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
        price_name = self.price_name
        state_lst = []
        for idx, _row in test_set.iterrows():

            state = _row.state
            state_lst.append(state)
            if pos == 0:
                if state in long_states:
                    pos = 1
                    cost = _row[price_name] * (1 + fee)
                    s_time = _row.date_time
                    hold_price = []
                    high_price = []
                    hold_price.append(cost / (1 + fee))
                    high_price.append(cost / (1 + fee))
                    high_price.append(_row.high)
                    chg = (pos * _row.close / cost + (1 - pos)) - 1
                elif state in short_states:
                    cost = _row[price_name] * (1 - fee)
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
                    s_price = _row[price_name] * (1 - fee)
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
                    s_price = _row[price_name] * (1 - fee)
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
                    b_price = _row[price_name] * (1 + fee)
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
                    b_price = _row[price_name] * (1 + fee)
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
        self.ret = ret
        net_lst = ret['net'].tolist()
        annR = annROR(net_lst, 1)
        sharp = yearsharpRatio(net_lst, 1)
        max_retrace = maxRetrace(net_lst, 1)
        self.annR = annR
        self.sharp = sharp
        self.max_retrace = max_retrace

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

    def get_netdf(self, i):
        data_set = self.data_set
        train_days = self.train_days
        test_days = self.test_days
        len_train = test_days
        if i + test_days >= len(self.data_set):
            all_set = data_set.iloc[i - train_days:]
            train_set = data_set.iloc[i - train_days:i]
            test_set = data_set.iloc[i:]
        else:
            all_set = data_set.iloc[i - train_days:i + test_days]
            train_set = data_set.iloc[i - train_days:i]
            test_set = data_set.iloc[i:i + test_days]
        model, hidden_states, long_states, short_states, random_states = self.get_longshort_state_from_cumsum(
            train_set)
        test_df = test_set[self.cols_features]

        test_df = test_df.assign(hidden_states=lambda df: [model.predict(df.iloc[get_index(j, len_train):j, :])[-1] for j in range(1, len(df) + 1)])
        # test_df = test_df.assign(
        #     hidden_states=lambda df: df.apply(lambda x: model.predict(np.array(x.rolling(len_train)))[-1]))
        hidden_states_predict = test_df['hidden_states'].tolist()

        if self.type == 'stock':
            ret, signal_state = self.handle_data_stock(test_set, hidden_states_predict, long_states, short_states,
                                                       random_states)
        else:
            ret, signal_state = self.handle_data(test_set, hidden_states_predict, long_states, short_states,
                                                 random_states)
        return ret


    def _get_ret_outsample_opt(self):
        self.insample = False
        ret_df = []
        signal_state_all = []
        train_days = self.train_days
        test_days = self.test_days
        for i in range(train_days, len(self.data_set), test_days):
            ret = self.get_netdf(i)
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
            if i + test_days >= len(self.data_set):
                all_set = self.data_set.iloc[i - train_days:]
                train_set = self.data_set.iloc[i - train_days:i]
                test_set = self.data_set.iloc[i:]
            else:
                all_set = self.data_set.iloc[i - train_days:i + test_days]
                train_set = self.data_set.iloc[i - train_days:i]
                test_set = self.data_set.iloc[i:i + test_days]
            model, hidden_states, long_states, short_states, random_states = self.get_longshort_state_from_cumsum(
                train_set)
            hidden_states_predict = []
            test_df = all_set[self.cols_features]
            len_train = len(train_set)
            len_test = len(test_set)
            for n in range(len(test_set)):
                hidden_states_predict.append(model.predict(test_df.head(len_train + n + 1).tail(len_test))[-1])
            # test_df = all_set[self.cols_features]
            # len_train_set = len(train_set)
            # for n in range(len(test_set)):
            #     hidden_states_predict.append(model.predict(test_df.head(len_train_set + n + 1))[-1])
            if self.type == 'stock':
                ret, signal_state = self.handle_data_stock(test_set, hidden_states_predict, long_states, short_states,
                                                     random_states)
            else:
                ret, signal_state = self.handle_data(test_set, hidden_states_predict, long_states, short_states,
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
        return self.annR, self.sharp, self.max_retrace, self.ret

    def run_outsample_opt(self):
        self._get_ret_outsample_opt()
        self._analyze()
        return self.annR, self.sharp, self.max_retrace, self.ret
