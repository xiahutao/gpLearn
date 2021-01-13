#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:19:53 2019
单因子分析 基于alphalens——进行因子有效性评分
@author: lion95
"""

from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import mstats
from scipy import stats
import datetime
# from CAL.PyCAL import *
import os
import alphalens
# alphalens
import matplotlib.pyplot as plt
import seaborn
import math
from jqdatasdk import *

auth('18610039264', '')


# from  pre_factors import *

def winsorize_series(se):
    q = se.quantile([0.025, 0.975])
    if isinstance(q, pd.Series) and len(q) == 2:
        se[se < q.iloc[0]] = q.iloc[0]
        se[se > q.iloc[1]] = q.iloc[1]
    return se


def standardize_series(se):
    se_std = se.std()
    se_mean = se.mean()
    return (se - se_mean) / se_std


def yearsharpRatio(netlist, n=240):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(365 * 1440 / n, 0.5)


if __name__ == '__main__':
    py_path = r'/Users/yeecall/Documents/mywork/joinquant_data/量价因子分析'

    stocks = get_index_stocks('000300.XSHG')
    ots = '_1d_2010-01-01_2019-05-01.csv'
    file_lst = [i + ots for i in stocks]
    os.chdir(py_path)

    # 数据加载
    a = list(range(1, 238))
    alpha_test = []
    for x in a:
        if x < 10:
            alpha_test.append("Alpha.alpha00" + str(x))
        elif 10 < x < 100:
            alpha_test.append("Alpha.alpha0" + str(x))
        else:
            alpha_test.append("Alpha.alpha" + str(x))
    #    alpha_test=['Alpha.alpha001']

    # 加载行情数据
    num = 0
    for f in file_lst:
        #        data_file=sy+'_4h_BIAN.csv'
        print(f)
        if num == 0:
            price = pd.read_csv('stk_price_data/' + f, index_col=0)
            price = price[['tradedate', 'close']]
            price['kind'] = f[:11]
        else:
            temp = pd.read_csv('stk_price_data/' + f, index_col=0)
            temp = temp[['tradedate', 'close']]
            temp['kind'] = f[:11]
            price = pd.concat([price, temp])
        num = num + 1
    price.columns = ['tradeDate', 'closePrice', 'secID']

    #    num=0
    symbols = stocks
    res_lst = list()
    for al in alpha_test:
        try:
            print(al)
            rst = list()
            factor_data = pd.DataFrame()
            num = 0
            for sy in symbols:
                file_name = sy + '_' + al + '.csv'
                print(file_name)
                if num == 0:
                    factor_data = pd.read_csv('factors_300/' + file_name, index_col=0)
                else:
                    temp = pd.read_csv('factors_300/' + file_name, index_col=0)
                    factor_data = pd.concat([factor_data, temp])
                num = num + 1
            data = factor_data.dropna()
            data.columns = ['tradeDate', 'secID', 'factor']

            tot = data.merge(price, on=['tradeDate', 'secID'])

            #            date_t_lst=tot.tradeDate.drop_duplicates().tolist()
            #            N=len(date_t_lst)
            #
            #            end=str(datetime.datetime.today())[:10]
            #            start=str(datetime.datetime.today()-datetime.timedelta(days=(N-1)))[:10]
            #            date_l=[datetime.datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start,end))]
            #
            #            food=dict(zip(date_t_lst,date_l))
            #            tot['tradeDate']=tot['tradeDate'].map(food)

            tot['tradeDate'] = pd.to_datetime(tot['tradeDate'], format='%Y-%m-%d')

            price_init = tot[['tradeDate', 'secID', 'closePrice']]
            price_init = price_init.pivot(index='tradeDate', columns='secID', values='closePrice')
            price_init = price_init.fillna(method='bfill')
            price_init = price_init.asfreq('C')
            #            print(price_init.index)

            factor_init = tot[['tradeDate', 'secID', 'factor']]
            factor_init = factor_init.pivot(index='tradeDate', columns='secID', values='factor')
            factor_init = factor_init.asfreq('C')
            factor_init = factor_init.stack()

            # 因子标准化
            #        factor_init = factor_init.groupby(level='tradeDate').apply(winsorize_series)     # 去极值
            #        factor_init = factor_init.groupby(level='tradeDate').apply(standardize_series)  # 标准化
            #        factor_init.hist(figsize=(12,6), bins=50)

            out = alphalens.utils.get_clean_factor_and_forward_returns(factor_init, price_init, quantiles=5,
                                                                       periods=[10, 20, 30])
            out = out.reset_index()
            out = out[['date', 'asset', '20D', 'factor', 'factor_quantile']]

            ic_lst = list()
            for idx, group_ in out.groupby('date'):
                #                print(idx)
                group_ = group_.dropna()
                ic_lst.append(group_[['20D', 'factor']].corr().iloc[0, 1])
            ic_m = np.nanmean(ic_lst)
            ic_d = np.nanstd(ic_lst)
            # 计算IR
            ir = ic_m / ic_d

            # 计算收益相关性大于0.6占比
            rev = out[['date', '20D', 'factor_quantile']]
            rev = rev.groupby(['date', 'factor_quantile']).mean().reset_index()
            corr_lst = list()
            for idx, group_ in rev.groupby('date'):
                #                print(idx)
                temp = group_.copy()
                temp['rank'] = temp['20D'].rank()
                corr_lst.append(temp[['rank', 'factor_quantile']].corr().iloc[0, 1])
            if ir > 0:
                over06 = [1 for i in corr_lst if i >= 0.6]
            else:
                over06 = [1 for i in corr_lst if i <= 0.6]
            corr_value = len(over06) / len(corr_lst)

            if ir > 0:
                temp = rev.query('factor_quantile==5')
                temp.index = range(len(temp))
                r_list = temp['20D'].tolist()
                r = list()
                for i in range(len(temp)):
                    if i % 20 == 0:
                        r.append(r_list[i])
                r_df = pd.DataFrame(r, columns=['20D'], index=range(len(r)))
                r_df = r_df.assign(add_one=lambda df: df['20D'] + 1) \
                    .assign(curve=lambda df: df.add_one.cumprod())
            else:
                temp = rev.query('factor_quantile==1')
                temp.index = range(len(temp))
                r_list = temp['20D'].tolist()
                r = list()
                for i in range(len(temp)):
                    if i % 20 == 0:
                        r.append(r_list[i])
                r_df = pd.DataFrame(r, columns=['20D'], index=range(len(r)))
                r_df = r_df.assign(add_one=lambda df: df['20D'] + 1) \
                    .assign(curve=lambda df: df.add_one.cumprod())
            sharp = yearsharpRatio(r_df.curve.tolist())
            rst.append(al)
            rst.append(sharp)
            rst.append(corr_value)
            rst.append(ir)
            res_lst.append(rst)
        except:
            print('{var} is loss'.format(var=al))
    result = pd.DataFrame(res_lst, columns=['name', 'sharp', 'corr_value', 'IR'])
    result.to_excel('result/factors_result_stk_3.xls')

    # 因子评分
    total = len(result)
    temp = result.copy()
    temp['IRs'] = temp.IR.apply(lambda s: abs(s))
    temp = temp.assign(sharp_rank=100 * (total - temp.sharp.rank(ascending=False) + 1) * 3 / total) \
        .assign(IR_ratio_rank=100 * (total - temp.IRs.rank(ascending=False) + 1) * 2 / total) \
        .assign(corr_value_rank=100 * (total - temp.corr_value.rank(ascending=False) + 1) / total) \
        .assign(rank=lambda df: df.sharp_rank + df.IR_ratio_rank + df.corr_value_rank)
    temp = temp.sort_values(by='rank', ascending=False)
    #
    #    print(result)
    temp.to_excel('result/factors_result_stk_20_3.xls')
