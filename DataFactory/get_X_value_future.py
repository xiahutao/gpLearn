# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/18 11:46
@Author  : zhangfang
@File    : get_X_value.py
'''
from __future__ import division
import pandas as pd
import numpy as np
import time
# from sqlalchemy import create_engine
import datetime
from DataFactory.factors201 import *
from jqdatasdk import *
from DataFactory.configDB import *
auth(JOINQUANT_USER, JOINQUANT_PW)


def get_stock_hq(sec, sdate, edate):
    df = pd.read_csv('c:/e/data/stock_hq/%s_daily.csv' % sec, encoding='gbk')
    df['date_time'] = df['date_time'].apply(lambda x: str(x)[:10])
    df = df[(df['date_time'] >= sdate) & (df['date_time'] <= edate)]
    df = df.reset_index(drop=True)
    return df


def get_future_hq(sec, sdate, edate):
    df = pd.read_csv('c:/e/data/future_index/%s_daily_index.csv' % sec, encoding='gbk')
    df['date_time'] = df['date_time'].apply(lambda x: str(x)[:10])
    df = df[(df['date_time'] >= sdate) & (df['date_time'] <= edate)]
    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    path = 'c:/Users/51951/PycharmProjects/gpLearn/'
    resualt_path = 'c:/e/data/factor/'
    future_period = 20
    sdate = '2014-01-01'
    edate = '2020-11-01'
    today = datetime.date.today()
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                  'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                  'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    # 获取所有因子列表
    df = pd.DataFrame([], columns=['open', 'high', 'low', 'close', 'volume', 'money'])
    # features_lst = Alphas(df).get_alpha_methods()
    # features_lst = features_lst[:5]
    features_lst = ['alpha801', 'alpha802', 'alpha803', 'alpha804', 'alpha805']
    indicator_df = []
    for code in symbol_lst:
        print(code)
        hq = get_future_hq(code, sdate, edate)
        Alpha = Alphas(hq)
        ret = hq[['date_time', 'close']]
        ret['future_ret'] = ret['close'].shift(-future_period) / ret['close'] - 1
        for feature in features_lst:
            ret[feature] = eval('Alpha.' + feature)()
        ret.to_csv(resualt_path + 'indicator_%s%s.csv' % (code, future_period), encoding='gbk')
        # indicator_df.append(ret)
    # indicator_df = pd.concat(indicator_df)
    # print(indicator_df.head(10))
    # print(indicator_df.tail(10))
    # print(len(indicator_df))
    # indicator_df.to_csv(resualt_path + 'stock_indicator.csv', encoding='gbk')