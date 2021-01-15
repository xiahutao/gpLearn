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


# 获取价格
def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


if __name__ == '__main__':
    path = 'c:/Users/51951/PycharmProjects/gpLearn/'
    resualt_path = 'c:/g/lfp/fund_factor/'
    future_period = 20
    sdate = '2011-01-01'
    edate = '2021-1-11'
    today = datetime.date.today()
    fund_value = pd.read_hdf(resualt_path + 'fund_value_402004.h5', 'all')\
        .rename(columns={'day': 'date_time'})
    benchmark = stock_price('000300.XSHG', '1d', sdate, today)
    benchmarknet = benchmark[['tradedate', 'close']].rename(columns=
                                                            {'tradedate': 'date_time', 'close': 'b_mark'})
    benchmarknet.date_time = benchmarknet.date_time.apply(lambda s: str(s)[:10])
    fund_value = fund_value[['code', 'date_time', 'sum_value']]
    fund_value.date_time = fund_value.date_time.apply(lambda s: str(s))
    fund_value_bench = pd.merge(fund_value, benchmarknet, on='date_time', how='left')
    lst = ['000196', '000894', '000904', '000992', '001117', '001224', '001274', '001387', '001388', '001428', '001443',
           '001444', '001522', '001523', '001543', '001570', '001574', '001580', '001585', '001607', '001608', '001613',
           '001615', '001620', '001635', '001636', '001641', '001664', '001665', '001681', '001683', '001686', '001687',
           '001688', '001716', '001728', '001734', '001735', '001740', '001761', '001762', '001763', '001769', '001772',
           '001773', '001774', '001791', '001792', '001795', '001796', '001801', '001808', '001822', '001837', '001838',
           '001858', '001861', '001866', '001869', '001890', '001892', '001897', '001903', '001904', '001905', '001908',
           '001910', '001922', '001924', '001951', '001959', '001967', '001979', '001985', '001997', '002009', '002010',
           '002015', '002016', '002018', '002019', '002020', '002025', '002026', '002027', '002046', '002085', '002087',
           '002088', '002137', '002141', '002145', '002148', '002156', '002157', '002158', '002159', '002162', '002163',
           '002165', '002166', '002167', '002170', '002172', '002178', '002182', '002186', '002197', '002207', '002213',
           '002220', '002224', '002231', '002232', '002233', '002249', '002252', '002271', '002281', '002293', '002296',
           '002339', '165527', '519177', '519644', '519960', '519962', '960001', '960002', '960004', '960005', '960006',
           '960007', '960008', '960010', '960011', '960012', '960016', '960017', '960018', '960020', '960023', '960024']
    # 获取所有因子列表
    df = pd.DataFrame([], columns=['open', 'high', 'low', 'close', 'volume', 'money'])
    features_lst = Alphas(df).get_fund_methods()
    # features_lst = features_lst[:5]
    # features_lst = ['alpha003']
    indicator_df = []
    er = []
    for code, group in fund_value_bench.groupby(['code']):
        print(code)
        if code not in lst:
            continue
        t0 = time.time()
        hq = group
        Alpha = Alphas(group)
        ret = group[['date_time', 'sum_value']]
        ret['future_ret'] = ret['sum_value'].shift(-future_period-1) / ret['sum_value'].shift(-1) - 1
        for feature in features_lst:
            try:
                ret[feature] = eval('Alpha.' + feature)()
            except Exception as e:
                print(str(e))
                ret[feature] = None
                er.append([code, feature])
        ret.to_hdf(resualt_path + 'fund_factor.h5', str(code))
        print(time.time() - t0)
    df = pd.DataFrame(er, columns=['code', 'feature'])
    df.to_csv(resualt_path + 'error.csv')
        # indicator_df.append(ret)
    # indicator_df = pd.concat(indicator_df)
    # print(indicator_df.head(10))
    # print(indicator_df.tail(10))
    # print(len(indicator_df))
    # indicator_df.to_csv(resualt_path + 'stock_indicator.csv', encoding='gbk')