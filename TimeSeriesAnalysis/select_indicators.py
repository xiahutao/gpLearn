# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:45:42 2020
对指标体系进行指标筛选
Step 1： 数据整理，分别计算每只基金的各种指标的月频指标，未来1年的MRAR（2）值以及基于未来1年的MRAR（2）值的目标评级；
Step 2： 计算各种指标的IC，IR值，根据筛选规则进行指标一次筛选；
Step 3： 计算各个指标的方差值和协相关矩阵，剔除常量指标和相关性大的指标；
Step 4 :  分别对每个指标值与目标评级结果进行卡方检验，剔除p>0.05的指标；
@author: zf
"""

from __future__ import division
import pandas as pd
# from MySQLdb import connect, cursors
# import configparser
# import socket
import numpy as np
# import time
# from sqlalchemy import create_engine  
from datetime import datetime
# from dateutil.relativedelta import relativedelta
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# 将日频数据转化为月频数据
def data_monthly(df):
    num = 0
    for idx, group_ in df.groupby(['InnerCode', 'M_O_Date']):
        print(idx)
        tmp = group_.head(1).copy()
        if num == 0:
            ret = tmp
        else:
            ret = pd.concat([ret, tmp])
        num += 1
    return ret


# 数据拼接
def data_connect(df1, df2, col, method):
    tmp = pd.merge(df1, df2, on=col, how=method)

    return tmp


# 计算IC值
def IC_cal(df, lst):
    col = lst
    num = 0
    for i in col:
        t = df.loc[:, ['M_O_Date', 'mrar2', i]].dropna()
        tmp = t.groupby('M_O_Date').apply(lambda df: df.corr().iloc[0, 1]) \
            .reset_index().rename(columns={0: i})
        if num == 0:
            ret = tmp
        else:
            ret = pd.merge(ret, tmp, on=['M_O_Date'], how='outer')
        num += 1

    return ret


# 计算IR值
def IR_cal(df):
    IR = df.set_index('M_O_Date').apply(lambda s: np.mean(s) / np.std(s))

    return IR


# 计算IC值　大于　阈值的比率
def IC_over_ratio(df, num):
    tmp = df.copy()

    tmp[tmp > num] = 1
    tmp[tmp <= num] = 0
    tmp = tmp.sum() / len(tmp)

    return tmp


# 按给定的IC，IR筛选条件筛选指标
def select_by_ic_ir(ir, ic_over):
    tmp = pd.concat([ir, ic_over], axis=1).rename(columns={0: 'IR', 1: 'IC_over'})
    tmp.IR = abs(tmp.IR)

    return tmp.query('IR > 0.5 and IC_over > 0.65').index.tolist()


# 剔除常量
def reject_constant(df, lst):
    tmp = df[lst].copy()
    selector = VarianceThreshold(0.00001)
    selector.fit(tmp)
    support = selector.get_support(True)

    return [lst[i] for i in support]


# 剔除相关性高的的指标
def reject_corr(df, ir, col_lst):
    lst = [i for i in col_lst]
    IR_abs = abs(ir)
    t = df[col_lst]
    corr = t.corr()
    for i in range(len(col_lst)):
        for j in range(i + 1, len(col_lst)):
            tmp = corr.iloc[i, j]
            if abs(tmp) > 0.85:
                try:
                    if IR_abs.loc[col_lst[i]] > IR_abs.loc[col_lst[j]]:
                        lst.remove(col_lst[j])
                    else:
                        lst.remove(col_lst[i])
                except:
                    pass
    return lst


# 剔除没有通过卡方检验的指标
def chi2_test(df, col_lst):
    lst = [i for i in col_lst]
    lst.append('stars')
    tmp = df[lst]
    tmp[col_lst] = (tmp[col_lst] - tmp[col_lst].min()) / (tmp[col_lst].max() - tmp[col_lst].min())
    model1 = SelectKBest(chi2, k=len(col_lst))
    model1.fit_transform(tmp[col_lst], tmp['stars'])
    p_lst = model1.pvalues_
    p_less_index = np.where(p_lst < 0.05)
    re_lst = [col_lst[i] for i in p_less_index[0].tolist()]

    return re_lst


# 按照有效性筛选的流程选择指标
def select_indicators(df, col_list):
    IC = IC_cal(df, col_list)
    IR = IR_cal(IC)
    IC_over = IC_over_ratio(abs(IC.iloc[:, 1:]), 0.1)
    select_ic_ir = select_by_ic_ir(IR, IC_over)
    mo_no_constant = reject_constant(df, select_ic_ir)
    rejectcorr = reject_corr(df, IR, mo_no_constant)
    chi2test = chi2_test(df, rejectcorr)

    return chi2test


if __name__ == '__main__':
    path = 'c:/Users/51951/PycharmProjects/gpLearn/'
    # 常量
    INDICATORS = ['annror_', 'maxretrace_', 'avgretrace_', 'var_', 'yearsharpRatio_', 'sortinoratio_', 'IR_']
    # 读取指标数据
    df = pd.read_csv(path + 'funds_indicators.csv', index_col=0)
    df = df.dropna()
    # 读取未来一年晨星评级数据
    ms_fund_mark = pd.read_csv(path + 'mrar_k_-12.csv', index_col=0)

    # 将日频率数据转化为月频率数据
    mo = data_monthly(df)
    mo['M_O_Date'] = mo['M_O_Date'].apply(lambda s: datetime.strptime(s, '%b-%y').strftime('%Y-%m'))
    # 数据拼接
    rank_and_indicator = data_connect(ms_fund_mark, mo, ['M_O_Date', 'InnerCode'], 'inner')

    # 计算 IC ，IR值
    IC = IC_cal(rank_and_indicator, INDICATORS)
    IR = IR_cal(IC)
    IC_over = IC_over_ratio(abs(IC.iloc[:, 1:]), 0.1)

    # 通过 IC，IR值筛选指标
    select_ic_ir = select_by_ic_ir(IR, IC_over)

    # 剔除常量 
    mo_no_constant = reject_constant(rank_and_indicator, INDICATORS)
    # 剔除相关性
    rejectcorr = reject_corr(rank_and_indicator, IR, INDICATORS)
    # 卡方检验
    chi2test = chi2_test(rank_and_indicator, INDICATORS)
    # 综合检验                
    final_indicators = select_indicators(rank_and_indicator, INDICATORS)
