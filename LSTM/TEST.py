# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/25 10:11
@Author  : zhangfang
@File    : TEST.py
'''
# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
import pandas as pd
from LSTM.backtest_func import yearsharpRatio, maxRetrace, annROR
import copy
import tensorflow
import keras
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
print(tensorflow.__version__)
print(keras.__version__)