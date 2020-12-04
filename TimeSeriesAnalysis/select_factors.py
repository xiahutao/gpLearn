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
import numpy as np
import time
# from sqlalchemy import create_engine  
import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from DataFactory.factors201 import *
import inspect
from jqdatasdk import *
from DataFactory.configDB import *
import os
auth(JOINQUANT_USER, JOINQUANT_PW)
from scipy import stats



def get_kafang_pvalue(df):
    '''
    :param df: 两列，n×2
    :return:
    '''
    ret = stats.chi2_contingency(observed=df)
    print(ret)
    return ret[1]


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


def get_cut_point(series):
    cut_points = [np.percentile(series, i) for i in [20, 40, 60, 80]]
    return cut_points


def get_star(x, cut_points):
    if x <= cut_points[0]:
        return 1
    for i in range(len(cut_points)-1):
        if x > cut_points[i] and x <= cut_points[i+1]:
            return i + 2
    if x > cut_points[-1]:
        return len(cut_points) + 1



def star_future_ret(df):
    tmp = copy.deepcopy(df)
    tmp = tmp.groupby('date_time').apply(lambda s: get_cut_point(s)).reset_index(drop=False)
    # cut_point = []
    # for date, group in tmp.groupby(['date_time']):
    #     print(date, len(group))
    #     print()
    #     cut_point.append([date, get_cut_point(group[['future_ret']])])
    # cut_point_df = pd.DataFrame(cut_point, columns=['date_time', 'cut_point'])

    tmp = tmp.rename(columns={0: 'cut_point'})
    print(tmp)
    return tmp


# 计算IC值
def IC_cal(df, feature_lst):
    num = 0
    for i in feature_lst:
        t = df.loc[:, ['date_time', 'future_ret', i]].dropna()
        tmp = t.groupby('date_time').apply(lambda df: df.corr().iloc[0, 1]) \
            .reset_index().rename(columns={0: i})
        if num == 0:
            ret = tmp
        else:
            ret = pd.merge(ret, tmp, on=['date_time'], how='outer')
        num += 1
    return ret


# 计算IR值
def IR_cal(df):
    IR = df.set_index('date_time').apply(lambda s: np.mean(s) / np.std(s))
    return IR


# 计算IC值　大于　阈值的比率
def IC_over_ratio(df, num):
    tmp = df.copy()

    tmp[tmp > num] = 1
    tmp[tmp <= num] = 0
    tmp = tmp.sum() / len(tmp)
    return tmp

# 计算IC值　大于　阈值的比率
def IC_mean(df):
    return abs(df.mean())


# 按给定的IC，IR筛选条件筛选指标
def select_by_ic_ir(ir, ic_over):
    tmp = pd.concat([ir, ic_over], axis=1).rename(columns={0: 'IR', 1: 'IC_over'})
    tmp.IR = abs(tmp.IR)

    return tmp.query('IR > 0.1 and IC_over > 0.01').index.tolist()


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
    for i in range(len(col_lst)-1):
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
    lst = copy.deepcopy(col_lst)
    lst.append('score')
    lst.append('date_time')
    colums_lst = copy.deepcopy(col_lst)
    colums_lst.append('date_time')
    tmp = df[lst]
    tmp[col_lst] = (tmp[col_lst] - tmp[col_lst].min()) / (tmp[col_lst].max() - tmp[col_lst].min())
    p_lst_all = []
    for date, group in tmp.groupby(['date_time']):
        # print(len(group))
        model1 = SelectKBest(chi2, k=len(col_lst))
        model1.fit_transform(group[col_lst], group['score'])
        p_lst = list(model1.pvalues_)
        p_lst.append(date)
        p_lst_all.append(p_lst)
    p_lst_df = pd.DataFrame(p_lst_all, columns=colums_lst)
    print(p_lst_df)
    p_lst = p_lst_df.set_index('date_time').mean()
    print(p_lst)
    p_less_index = np.where(p_lst < 0.05)
    re_lst = [col_lst[i] for i in p_less_index[0].tolist()]
    return re_lst


# 按照有效性筛选的流程选择指标
def select_indicators(df, col_list):
    IC = IC_cal(df, col_list)
    IR = IR_cal(IC)
    # IC_over = IC_over_ratio(abs(IC.iloc[:, 1:]), 0.1)
    ic_mean = IC_mean(IC.iloc[:, 1:])
    select_ic_ir = select_by_ic_ir(IR, ic_mean)
    mo_no_constant = reject_constant(df, select_ic_ir)
    rejectcorr = reject_corr(df, IR, mo_no_constant)
    chi2test = chi2_test(df, rejectcorr)
    return chi2test


def get_stock_hq(sec, sdate, edate):
    df = pd.read_csv('c:/e/data/stock_hq/%s_daily.csv' % sec, encoding='gbk')
    df['date_time'] = df['date_time'].apply(lambda x: str(x)[:10])
    df = df[(df['date_time'] >= sdate) & (df['date_time'] <= edate)]
    return df


if __name__ == '__main__':
    path = 'c:/Users/51951/PycharmProjects/gpLearn/'
    resualt_path = 'c:/e/data/factor/'
    # 获取所有因子列表
    sdate = '2014-01-01'
    edate = '2020-01-01'
    today = datetime.date.today()
    listday_info = get_all_securities(types=['stock']).query("start_date<'{date}'".format(date=sdate))
    stock_lst = listday_info.index.tolist()
    df = pd.DataFrame([], columns=['open', 'high', 'low', 'close', 'volume', 'money'])
    features_lst = Alphas(df).get_alpha_methods()
    # features_lst = features_lst[:5]
    colums_lst = copy.deepcopy(features_lst)
    colums_lst = ["alpha046", "alpha049", "alpha052", "alpha071", "alpha082", "alpha093", "alpha096", "alpha112",
                  "alpha128", "alpha129", "alpha133"]
    colums_lst.extend(['date_time', 'close', 'future_ret'])
    indicator_df = []
    for code in stock_lst:
        ret = pd.read_csv(resualt_path + 'indicator_%s.csv' % code[:6], encoding='gbk', index_col=0)[colums_lst]
        ret['symbol'] = code
        indicator_df.append(ret)
    indicator_df = pd.concat(indicator_df)
    print(len(indicator_df))
    indicator_df = indicator_df.replace([np.inf, -np.inf], np.nan)

    print(indicator_df[indicator_df['date_time'] == '2020-10-19'])

    # 计算 IC ，IR值
    IC = IC_cal(indicator_df, features_lst)
    IR = IR_cal(IC)

    # 计算IC值大于某一阈值的比率
    ic_mean = IC_mean(IC.iloc[:, 1:])
    print('ic_mean')
    ic_mean_df = pd.DataFrame(ic_mean, columns=['IC'])
    print(ic_mean_df)
    print(ic_mean_df.sort_values(['IC'], ascending=False).head(10))
    print('IR')
    IR_df = pd.DataFrame(IR, columns=['IR'])
    print(IR_df.sort_values(['IR'], ascending=False).head(10))

    # 通过 IC，IR值筛选指标
    select_ic_ir = select_by_ic_ir(IR, ic_mean)
    print('IC-IR:', select_ic_ir)

    # 剔除常量 
    mo_no_constant = reject_constant(indicator_df, features_lst)
    print('剔除常量:', mo_no_constant)
    # 剔除相关性
    star = star_future_ret(indicator_df[['date_time', 'future_ret']])

    indicator_df = indicator_df.merge(star, on='date_time')
    future_ret = np.array(indicator_df['future_ret'])
    cut_point = np.array(indicator_df['cut_point'])
    score = [get_star(x, y) for x, y in zip(future_ret, cut_point)]
    indicator_df['score'] = score
    a = indicator_df[['score', 'future_ret', 'cut_point']]
    rejectcorr = reject_corr(indicator_df, IR, features_lst)
    # 卡方检验
    chi2test = chi2_test(indicator_df, features_lst)

    # 综合检验                
    final_indicators = select_indicators(indicator_df, features_lst)
