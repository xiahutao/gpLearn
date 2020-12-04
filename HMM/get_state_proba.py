# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Users\\51951\\PycharmProjects')  # 新加入的
sys.path.append('C:\\Users\\51951\\PycharmProjects\\joinquant')  # 新加入的
from HMM.HMM_function import *
import numpy as np
from matplotlib import cm, pyplot as plt
import seaborn as sns
import pandas as pd
from HMM.backtest_func import yearsharpRatio, maxRetrace, annROR
from jqdatasdk import *
from DataFactory.configDB import *

auth(JOINQUANT_USER, JOINQUANT_PW)
import warnings
import time

warnings.filterwarnings("ignore")


def get_dataset_close(data_ori, factor_lst):
    dataset = data_ori.loc[:, ['open', 'high', 'low', 'close', 'volume']]
    for alpha in factor_lst:
        alpha = 'Alpha.' + alpha
        Alpha = Alphas(dataset)
        dataset[alpha[-8:]] = eval(alpha)()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = pd.concat(
        [dataset[factor_lst], data_ori[['high', 'low', 'close', 'volume', 'open', 'date_time', 'close_1']]],
        axis=1)
    # 日涨跌幅，模型训练的Y值
    dataset['ret'] = dataset['close'].shift(-1) / dataset['close'] - 1
    dataset['ret'] = dataset['ret'].fillna(0)
    dataset = dataset.dropna()
    return dataset


if __name__ == '__main__':
    data_path = 'C:/e/data/stock_hq/'
    resualt_path = 'c:/e/hmm/resualt/future/'
    future_period = 1

    max_states = 6
    asset = '600519.XSHG'
    factor_lst = ['alpha000']
    s_data_date = '2010-01-01'
    e_data_date = '2020-12-01'
    all_feature_lst = PowerSetsRecursive(factor_lst)
    all_feature_lst = [i for i in all_feature_lst if len(i) >= 1]
    # all_feature_lst = [['price_deviation']]
    # # 三大股指
    ### 样本外不同滚动窗口回测，参数敏感性测试

    s_date = '2012-01-01'
    train_s_date = '2010-01-01'
    date_lst = [('2012-01-01', '2013-01-01'), ('2013-01-01', '2014-01-01'), ('2014-01-01', '2015-01-01'),
                ('2015-01-01', '2016-01-01'), ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01'),
                ('2018-01-01', '2019-01-01'), ('2019-01-01', '2020-01-01'), ('2020-01-01', '2021-01-01'), ]

    # all_feature_lst = [['price_deviation']]
    train_days_lst = [i for i in range(480, 50, -40)]
    test_days_lst = [i for i in range(20, 250, 20)]
    period_param_state = []
    asset_lst = ['A', 'AG', 'AL', 'AP', 'AU', 'B', 'BU', 'C', 'CF', 'CS', 'CU', 'FG',
                 'HC', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI', 'OI', 'P',
                 'PB', 'PP', 'RB', 'RM', 'RU', 'SC', 'SF', 'SM', 'SN', 'SR',
                 'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN', ['000931.XSHG', '000932.XSHG']]
    asset_lst = ['IF']
    train_days = 240
    test_days = 60
    for asset in asset_lst:
        t0 = time.time()
        # data_ori = pd.read_csv(data_path + asset + '_daily_index.csv')
        for feature_lst in all_feature_lst:


            select = pd.read_csv(data_path + '000931.XSHG' + '_daily.csv')[['date_time', 'close', 'open', 'high', 'low', 'volume', 'money']]
            select['alpha1'] = select['close'].rolling(10).apply(values_deviation, raw=True)
            select_no = pd.read_csv(data_path + '000932.XSHG' + '_daily.csv')[['date_time', 'close']]
            select_no['alpha2'] = select_no['close'].rolling(10).apply(values_deviation, raw=True)

            dataset = select[['date_time', 'alpha1']].merge(select_no[['date_time', 'alpha2']], on=['date_time'])
            print(dataset)
            # dataFactory = DataFactory(asset, factor_lst, future_period, s_data_date, e_data_date, asset)
            # dataset = dataFactory.get_dataset_close()
            dataset.index = dataset['date_time']
            data_set = dataset[(dataset.index >= train_s_date) & (dataset.index <= '2020-04-21')]

            # strategy_obj = HmmStrategy(
            #     asset=asset, s_date=s_date, max_states=max_states, leverage=1, fee=0.001, data_set=data_set,
            #     cols_features=feature_lst, price_name='close', type='future', train_days=train_days,
            #     test_days=test_days)
            data_set['date_time'] = pd.to_datetime(data_set['date_time'])
            net_new = copy.deepcopy(data_set).dropna()
            feature_lst = ['alpha1', 'alpha2']
            X = np.column_stack([net_new.loc[:, feature_lst]])
            # X = X.reshape(-1, 1)
            model = GaussianHMM(n_components=6, random_state=100, covariance_type='diag',
                        n_iter=10000)
            model = model.fit(X)
            # strategy_obj.get_state_predict_proba()
            predic = model.predict_proba(X)
            predic_df = pd.DataFrame(predic, columns=['state%s' %i for i in range(6)])
            net_new = net_new.reset_index(drop=True)
            print(net_new)
            net_new = pd.concat([net_new, predic_df], axis=1)
            print(net_new)
            plt.figure(figsize=(20, 10))
            net_new = net_new.set_index(['date_time'])
            net_new = net_new.loc['2020-01-01':'2020-04-21']
            # plt.legend(net_new[['state%s' % i for i in range(6)]], ['state%s' % i for i in range(6)], loc=0)

            # plt.plot(net_new[['state%s' %i for i in range(6)]])
            net_new[['state%s' %i for i in range(6)]].plot()
            # plt.legend(net_new[['state%s' % i for i in range(6)]], ['state%s' % i for i in range(6)])
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.title('样本内隐状态概率图')

            # plt.legend(0)

            plt.show()

            # print(asset, '特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days, annR, sharp, max_retrace)
            # for (sdate, edate) in date_lst:
            #     net = ret_df.loc[sdate:edate, :]
            #     if len(net) > 10:
            #         net_lst = net['net'].tolist()
            #         annR = annROR(net_lst, 1)
            #         sharp = yearsharpRatio(net_lst, 1)
            #         max_retrace = maxRetrace(net_lst, 1)
            #         period_param_state.append(
            #             [asset, sdate, edate, feature_lst, train_days, test_days, annR, sharp, max_retrace])

        print('================================================================', time.time() - t0)
    # period_param_state_df = pd.DataFrame(
    #     period_param_state, columns=['instrument', 'sdate', 'edate', '特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])
    # period_param_state_df.to_csv(resualt_path + 'future_sharp_diff_param_close.csv', encoding='gbk')
