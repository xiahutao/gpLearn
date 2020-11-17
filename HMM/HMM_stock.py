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
    future_period = 1
    max_states = 6
    asset = '600519.XSHG'
    factor_lst = ['alpha000']
    s_data_date = '2010-01-01'
    e_data_date = '2020-11-01'
    column_price = 'close'
    column_high = 'high'
    column_low = 'low'
    column_volume = 'volume'

    s_date = '2015-01-01'
    train_s_date = '2014-01-01'

    all_feature_lst = PowerSetsRecursive(factor_lst)
    all_feature_lst = [i for i in all_feature_lst if len(i) >= 1]
    # all_feature_lst = [['price_deviation']]
    train_days_lst = [240]
    test_days_lst = [60]
    period_param_state = []
    idx_code = '000985.XSHG'

    symbol_lst = index_stocks(idx_code)
    asset_lst = normalize_code(symbol_lst)
    # asset_lst = ['601166.XSHG']
    for feature_lst in all_feature_lst:
        for train_days in train_days_lst:
            for test_days in test_days_lst:
                for asset in asset_lst:
                    try:
                        data_ori = pd.read_csv(data_path + asset + '_' + 'daily.csv')
                    except:
                        data_ori = stock_price(asset, 'daily', '2010-01-01', '2020-12-30')
                    try:
                        dataFactory = DataFactory(asset, factor_lst, future_period, s_data_date, e_data_date, data_ori)
                        dataset = dataFactory.get_dataset_close()
                        data_set = dataset[dataset.index >= train_s_date]
                        print(asset, '特征因子：', feature_lst, '训练窗口:', train_days, '测试窗口:', test_days)
                        strategy_obj = HmmStrategy(
                            asset=asset, s_date=s_date, max_states=max_states, leverage=1, fee=0.001, data_set=data_set,
                            cols_features=feature_lst, price_name='close', type='stock', train_days=train_days, test_days=test_days,
                            insample=False)
                        annR, sharp, max_retrace, ret_df = strategy_obj.run_outsample()
                        f = plt.figure(figsize=(15, 8))
                        plt.rcParams['font.sans-serif'] = ['SimHei']

                        sharp_close = yearsharpRatio(ret_df.net_close.tolist(), 1)

                        # Plot return
                        # ax1 = f.add_subplot(111)
                        # ax1.plot(ret_df.net, 'blue', label='net_hmm')
                        # ax1.plot(ret_df.net_close, 'red', label='net_close')
                        # ax1.legend()
                        # ax1.set_title(
                        #     '%s特征因子%s训练窗口%s测试窗口%s净值曲线' % (asset, feature_lst, train_days, test_days))
                        # ax1.set_xlabel('Time')
                        # ax1.set_ylabel('Value')
                        # plt.show()
                        print('夏普比率: ' + str(np.around(sharp, 2))+'行情夏普比率: ' + str(np.around(sharp_close, 2))+
                              '年化收益: ' + str(np.around(annR, 2))+'最大回撤: ' + str(np.around(max_retrace, 2)))
                        period_param_state.append([
                            asset, feature_lst, train_days, test_days, annR, sharp, max_retrace, sharp_close])
                    except:
                        continue
    period_param_state_df = pd.DataFrame(
        period_param_state, columns=['instrument', '特征因子', '训练窗口', '测试窗口', '年化收益', '夏普比率', '最大回撤', '行情夏普'])
    print(period_param_state_df)
    period_param_state_df.to_csv('c:/e/hmm/resualt/stock_sharp_20150101_20201111_close_%s.csv' %idx_code, encoding='gbk')