# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/17 17:06
@Author  : zhangfang
@File    : kafang.py
'''
from scipy import stats
import pandas as pd
import numpy as np

def get_kafang_pvalue(df):
    '''
    :param df: 两列，n×2
    :return:
    '''
    ret = stats.chi2_contingency(observed=df)
    print(ret)
    return ret[1]


if __name__ == '__main__':
    data_path = 'c:/e/data/future_index/'
    df_rb = pd.read_csv('%sRB_daily_index.csv' % data_path)
    print(df_rb)
    df_rb['ret'] = df_rb['close'].shift(-1) / df_rb['close']
    df_rb['test'] = df_rb['close'].rolling(5).mean()
    print(df_rb)
    df_rb = df_rb.dropna()
    print(df_rb)
    a = get_kafang_pvalue(df_rb[['test', 'ret']])
    print(a)

    get_kafang_pvalue(np.array([[37, 49, 23], [150, 100, 57]]))
