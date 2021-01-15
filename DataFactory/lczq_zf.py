# -*- coding: utf-8 -*-
"""
联创证券面试题
"""
from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from jqdatasdk import *
auth('18610039264', 'zg19491001')


# 净值转化为日收益率
def price_to_rev(netlist):
    r = list()
    for i in range(1, len(netlist)):
        r.append(math.log(netlist[i] / netlist[i - 1]))
    return r

# 计算半方差
def Semivariance(x):
    r = price_to_rev(x)
    r_mean = np.mean(r)
    lows = [i for i in r if i <= r_mean]
    return np.sum((lows-r_mean)**2) / len(lows)
# 计算VaR 
def var(netlist, a=0.05):
    '''
    :param list:netlist
    :return: 95%置信概率的日VAR值
    '''
    r = price_to_rev(netlist)
    r_s = pd.Series(r)
    # r_s_p = r_s.rolling(period).apply(np.sum, raw=True)
    r_s = r_s.dropna()
    var = np.quantile(r_s, a, interpolation='linear')
    return (var)
# 计算IR
def ir(netlist1, netlist2):
    r1 = price_to_rev(netlist1)
    asset_return = pd.Series(r1)
    r2 = price_to_rev(netlist2)
    index_return = pd.Series(r2)
    multiplier = 252

    if asset_return is not None and index_return is not None:

        active_return = asset_return - index_return
        tracking_error = (active_return.std(ddof=1)) * np.sqrt(multiplier)

        asset_annualized_return = multiplier * asset_return.mean()
        index_annualized_return = multiplier * index_return.mean()

        information_ratio = (asset_annualized_return - index_annualized_return) / tracking_error

    else:
        information_ratio = np.nan

    return information_ratio
    
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
    # 读取数据
    df = pd.read_excel("基金净值数据.xlsx",encoding='gbk')
    fund_values = df[['日期','复权净值(元)']]
    fund_values.columns = ['tradedate','nv']
    # 计算业绩基准
    r_f = (1.75 / 100) / 252
    # 获取指数数据
    # 沪深300
    hs300 = stock_price('000300.XSHG', '1d', '2010-01-01', '2020-12-22')
    hs300['chg_hs300'] = hs300.close.diff() / hs300.close.shift(1)
    # 国债指数
    gz = stock_price('000012.XSHG', '1d', '2010-01-01', '2020-12-22')
    gz['chg_gz'] = gz.close.diff() / gz.close.shift(1)
    # 业绩基准
    b_mark = hs300[['tradedate','chg_hs300']].merge(gz[['tradedate','chg_gz']])
    b_mark['chg'] = 0.6 * b_mark['chg_hs300'] + 0.35 * b_mark['chg_gz'] + 0.05 * r_f
    b_mark.fillna(0,inplace=True)
    b_mark['benchmark'] = (b_mark['chg'] + 1).cumprod()
    # 计算相关指标
    b_mark.tradedate = b_mark.tradedate.apply(lambda s: str(s)[:10])
    value_data = fund_values.merge(b_mark)
    # 计算半方差
    Semivar = Semivariance(value_data.nv)
    # 计算VAR
    var_1 =  var(value_data.nv, a=0.05)
    # 计算IR值
    IR = ir(value_data.nv, value_data.benchmark)
    # HM 
    rf_hm = 0.0175/12
    value_data_2 = value_data[['tradedate','nv','chg']]
    value_data_2['nv_chg'] = value_data_2.nv.diff() / value_data_2.nv.shift(1)
    value_data_2.dropna(inplace=True)
    value_data_2['month'] = value_data_2.tradedate.apply(lambda s: str(s)[:7])
    hm_data = value_data_2[['month','nv_chg','chg']].groupby('month').apply(lambda x : (1 + x).prod() - 1)
    hm_data = hm_data.reset_index()
    tmp_x = hm_data.copy()
    tmp_x['y'] = tmp_x['nv_chg'] - rf_hm
    tmp_x['x1'] = tmp_x['chg'] - rf_hm
    tmp_x['x2'] = tmp_x['x1'].apply(lambda s: max(s, 0))
    x_tm = tmp_x[['x1', 'x2']]
    x_tm = sm.add_constant(x_tm)
    model_tm = sm.OLS(tmp_x['y'], x_tm).fit()
    [alpha_tm, beta1_tm, beta2_tm] = model_tm.params
    [p1_tm, p2_tm, p3_tm] = model_tm.pvalues
    # NM 模型结果
    model_tm.summary2()
