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

if __name__ == '__main__':
    data_path = 'C:/e/data/future_index/'
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
                 'T', 'TA', 'TF', 'V', 'Y', 'ZC', 'ZN']
    asset_lst = ['A', 'AG', 'AL']
    for asset in asset_lst:
        t0 = time.time()
        data_ori = pd.read_csv(data_path + asset + '_daily_index.csv')
        for feature_lst in all_feature_lst:
            for train_days in train_days_lst:
                for test_days in test_days_lst:
                    try:
                        dataFactory = DataFactory(asset, factor_lst, future_period, s_data_date, e_data_date, data_ori)
                        dataset = dataFactory.get_dataset_close()
                        data_set = dataset[dataset.index >= train_s_date]

                        strategy_obj = HmmStrategy(
                            asset=asset, s_date=s_date, max_states=max_states, leverage=1, fee=0.001, data_set=data_set,
                            cols_features=feature_lst, price_name='close', type='future', train_days=train_days,
                            test_days=test_days)
                        annR, sharp, max_retrace, ret_df = strategy_obj.run_outsample()
                        ret_df.to_csv(resualt_path + 'net_param/%s_%s_%s.csv' % (asset, train_days, test_days),
                                      encoding='gbk')
                        print(asset, '特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days, annR, sharp, max_retrace)
                        # for (sdate, edate) in date_lst:
                        #     net = ret_df.loc[sdate:edate, :]
                        #     if len(net) > 10:
                        #         net_lst = net['net'].tolist()
                        #         annR = annROR(net_lst, 1)
                        #         sharp = yearsharpRatio(net_lst, 1)
                        #         max_retrace = maxRetrace(net_lst, 1)
                        #         period_param_state.append(
                        #             [asset, sdate, edate, feature_lst, train_days, test_days, annR, sharp, max_retrace])
                    except Exception as e:
                        print(str(e))
                        continue
        print('================================================================', time.time() - t0)
    # period_param_state_df = pd.DataFrame(
    #     period_param_state, columns=['instrument', 'sdate', 'edate', '特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])
    # period_param_state_df.to_csv(resualt_path + 'future_sharp_diff_param_close.csv', encoding='gbk')
