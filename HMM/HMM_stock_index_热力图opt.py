# -*- coding: utf-8 -*-
from gpLearn.HMM.HMM_function import *

import numpy as np
from matplotlib import cm, pyplot as plt
from get_data.get_stock_hq import stock_price, index_stocks
import seaborn as sns
import pandas as pd
from backtest_func import yearsharpRatio, maxRetrace, annROR
from jqdatasdk import *
from configDB import *
auth(JOINQUANT_USER, JOINQUANT_PW)
import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    data_path = 'C:/e/data/stock_hq/'
    resualt_path = 'c:/e/hmm/resualt/stockindex/'
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
    train_days_lst = [i for i in range(480, 30, -40)]
    test_days_lst = [i for i in range(20, 250, 20)]
    asset_lst = ['000300.XSHG', '000016.XSHG', '000905.XSHG', '399006.XSHE']

    # period_param_state_df = pd.DataFrame(
    #     period_param_state, columns=['instrument', 'sdate', 'edate', '特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤'])
    period_param_state_df = pd.read_csv(resualt_path + 'stockindex_sharp_diff_param_close.csv', encoding='gbk')
    lst = []
    for feature_lst in all_feature_lst:
        for asset in asset_lst:
            for (sdate, edate) in date_lst:
                group_ = period_param_state_df[
                    (period_param_state_df['sdate'] == sdate) & (period_param_state_df['edate'] == edate) & (
                            period_param_state_df['instrument'] == asset)]
                print(group_)

                if len(group_) == 0:
                    continue
                best_sharp = 0
                train_days_best = 240
                test_days_best = 60
                for i in range(1, len(train_days_lst)-1):
                    train_days = train_days_lst[i]
                    for j in range(1, len(test_days_lst)-1):
                        test_days = test_days_lst[j]
                        sharp_lst = []
                        for m in range(i-1, i+2):
                            temp_train_days = train_days_lst[m]
                            for n in range(j-1, j+2):
                                temp_test_days = test_days_lst[n]
                                try:
                                    sharp = group_[
                                        (group_['测试窗口'] == temp_test_days)&(group_['训练窗口'] == temp_train_days)]['夏普比率'].tolist()[0]
                                except Exception as e:
                                    print(str(e))
                                    sharp = -1
                                sharp_lst.append(sharp)
                        if np.mean(sharp_lst) > best_sharp:
                            best_sharp = np.mean(sharp_lst)
                            train_days_best = train_days
                            test_days_best = test_days
                lst.append([feature_lst, asset, sdate, edate, train_days_best, test_days_best])
    ret = pd.DataFrame(lst, columns=['feature', 'asset', 's_date', 'e_date', 'train_days', 'test_days'])
    print(ret)
    ret.to_csv(resualt_path + 'para_opt_history.csv')



