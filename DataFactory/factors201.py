import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from functools import reduce
import warnings
import copy
import math
import statsmodels.api as sm
from pyfinance.ols import PandasRollingOLS

warnings.filterwarnings("ignore")


# data=pd.read_csv("D:\\workdata\\data\\btcusdt_1d.csv",index_col="Unnamed: 0")
# print(data.head())


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def max_s(x, y):
    value_list = [a if a > b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="max")


def min_s(x, y):
    value_list = [a if a < b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="min")


def sma(df, window=10):
    return df.rolling(window).mean()


def kurtosis(df, window=10):
    return df.rolling(window).kurt()


def skewness(df, window=10):
    return df.rolling(window).skew()


def stddev(df, window=10):
    return df.rolling(window).std()


def correlation(x, y, window=10):
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    return x.rolling(window).cov(y)


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    return window + 1 - df.rolling(window).apply(rolling_rank)


# 计算正收益率占比
def positive_ratio(df, window=10):
    return df.rolling(window).apply(rolling_positive_ratio)


def rolling_positive_ratio(r):
    p = [i for i in r if i > 0]
    return (len(p) / len(r))


def rolling_prod(na):
    return na.prod(na)


def product(df, window=10):
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def ts_count(x, y, window=10):
    diff = y - x
    diff[diff < 0] = np.nan
    result = diff.rolling(window).count()
    result[:window - 1] = np.nan
    return result


def delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def ranks(df):
    # print(df.rank(pct=True))
    return df.rank(pct=True).values[-1]


def rank(df, window=10):
    return df.rolling(window).apply(ranks, raw=False)


def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1


# 年化波动率
def rolling_volatility(lst):
    return np.std(lst, ddof=1) * math.pow(252, 0.5)


def ts_volatility(df, window=10):
    return df.rolling(window).apply(rolling_volatility)


# 最大回撤
def rolling_maxretrace(lst):
    running_max = np.maximum.accumulate(lst)
    underwater = (running_max - lst) / running_max
    return underwater.max()


def ts_maxretrace(lst):
    running_max = np.maximum.accumulate(lst)
    underwater = (lst - running_max) / running_max
    underwater = np.minimum.accumulate(underwater)
    return -underwater


def ts_lowday(df, window=10):
    return (window - 1) - df.rolling(window).apply(np.argmin)


def ts_highday(df, window=10):
    return (window - 1) - df.rolling(window).apply(np.argmax)


def rolling_avgretrace(lst):
    maxretrace = ts_maxretrace(lst)
    prod = len(lst) - 1 - np.array(range(len(lst)))
    retrace = 0.9 ** prod * maxretrace
    return (np.sum(retrace))


# 高水位回撤率
def ts_avgretrace(lst):
    maxretrace = ts_maxretrace(lst)
    ret = []
    for i in range(len(maxretrace)):
        maxretrace_ = maxretrace[:i + 1]
        lst_ = lst[:i + 1]
        ret.append(rolling_avgretrace(lst_, maxretrace_))
    return ret


# 基于历史法计算Var
def rolling_var(log_lst, a=0.01):
    r_s = pd.Series(log_lst)
    r_s = r_s.dropna()
    return np.quantile(r_s, a, interpolation='linear')


def SMA(vals, n, m):
    # 算法1
    return reduce(lambda x, y: ((n - m) * x + y * m) / n, vals)


def sma_list(df, n, m):
    result_list = [np.nan]
    for x in range(1, len(df)):
        if df.values[x - 1] * 0 == 0:
            value = SMA([df.values[x - 1], df.values[x]], n, m)
            result_list.append(value)
        elif df.values[x - 1] * 0 != 0:
            result_list.append(np.nan)
        elif df.values[x - 2] * 0 != 0:
            value = SMA([df.values[x - 1], df.values[x]], n, m)
            result_list.append(value)
        else:
            value = SMA([result_list[-1], df.values[x]], n, m)
            result_list.append(value)
    result_series = pd.Series(result_list, name="sma")
    return result_series


def decay_linear(df, period=10):
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)

    na_lwma = df.values
    y = list(range(1, period + 1))
    y.reverse()
    y = np.array(y)
    y = y / y.sum()
    value_list = [np.nan] * (period - 1)
    for pos in range(period, len(na_lwma)):
        value = na_lwma[pos - period:pos]
        value = value * y
        value_list.append(value.sum())
    return pd.Series(value_list, name="close")


def vol_estimator_garch(data_df, st=25, lt=252 * 3):  # 250*5):
    st_span = st  # min(st,len(data_df))
    lt_span = lt  # min(lt,len(data_df))
    # print(st_span, lt_span,st)
    st_vol = data_df.ewm(span=st_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
    lt_vol = data_df.ewm(span=lt_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
    decay_rate = 0.8
    vol = st_vol * decay_rate + lt_vol * (1 - decay_rate)
    # vol=self.cap_vol_by_rolling(vol)
    return vol


# 夏普比率
def yearsharpRatio(log_ret):
    asset_return = pd.Series(log_ret).dropna()
    annualized_return = 252 * asset_return.mean()
    annualized_vol = asset_return.std(ddof=1) * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.03) / annualized_vol
    return sharpe_ratio


# 索提诺比率
def sortinoratio(log_ret):
    target_return = log(pow(1.03, 1 / 252))
    asset_return = pd.Series(log_ret).dropna()
    downside_return = asset_return - target_return
    downside_return[downside_return > 0] = 0
    annualized_return = 252 * asset_return.mean()
    annualized_vol = downside_return.std(ddof=1) * np.sqrt(252)
    sortino_ratio = (annualized_return - 0.03) / annualized_vol
    return sortino_ratio


def get_resp_curve(x, method):
    resp_curve = pd.DataFrame()
    if method == 'gaussian':
        resp_curve = np.exp(-(x ** 2) / 4.0)
    return resp_curve


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.benchmark = []
        self.log_ret = []
        self.log_benchmark = []
        self.ex_log_ret = []
        # self.log_ret = []
        try:
            self.open = pn_data['open']
            self.high = pn_data['high']
            self.low = pn_data['low']
            self.close = pn_data['close']
            self.volume = pn_data['volume']
            self.amount = pn_data['money']
            self.returns = self.close - self.close.shift(1)
        except:
            self.close = pn_data['sum_value']
            self.benchmark = pn_data['b_mark']
            pn_data['log_ret'] = pd.Series(np.log(self.close))
            pn_data['log_ret'] = pn_data['log_ret'].diff()
            pn_data['log_benchmark'] = pd.Series(np.log(self.benchmark))
            pn_data['log_benchmark'] = pn_data['log_benchmark'].diff()
            pn_data = pn_data.fillna(0)
            self.log_ret = pn_data['log_ret']
            self.log_benchmark = pn_data['log_benchmark']
            self.ex_log_ret = self.log_ret - self.log_benchmark

    # fund001-005,半年、年、2年、3年、5年年化收益
    def fund001(self):
        return (self.close / self.close.shift(126)) ** 2 - 1

    def fund002(self):
        return self.close / self.close.shift(252) - 1

    def fund003(self):
        return (self.close / self.close.shift(504)) ** (1 / 2) - 1

    def fund004(self):
        return (self.close / self.close.shift(252 * 3)) ** (1 / 3) - 1

    def fund005(self):
        return (self.close / self.close.shift(252 * 5)) ** (1 / 5) - 1

    # fund006-010,半年、年、2年、3年、5年年化超额收益
    def fund006(self):
        return (self.close / self.close.shift(126)) ** 2 - (self.benchmark / self.benchmark.shift(126)) ** 2

    def fund007(self):
        return self.close / self.close.shift(252) - self.benchmark / self.benchmark.shift(252)

    def fund008(self):
        return (self.close / self.close.shift(504)) ** (1 / 2) - (self.benchmark / self.benchmark.shift(504)) ** (1 / 2)

    def fund009(self):
        return (self.close / self.close.shift(252 * 3)) ** (1 / 3) - (
                self.benchmark / self.benchmark.shift(252 * 3)) ** (1 / 3)

    def fund010(self):
        return (self.close / self.close.shift(252 * 5)) ** (1 / 5) - (
                self.benchmark / self.benchmark.shift(252 * 5)) ** (1 / 5)

    # fund011-015 半年、年、2年、3年、5年对数收益的峰度
    def fund011(self):
        return kurtosis(self.log_ret, 126)

    def fund012(self):
        return kurtosis(self.log_ret, 252)

    def fund013(self):
        return kurtosis(self.log_ret, 252 * 2)

    def fund014(self):
        return kurtosis(self.log_ret, 252 * 3)

    def fund015(self):
        return kurtosis(self.log_ret, 252 * 5)

    # fund016-020 半年、年、2年、3年、5年超额对数收益的峰度
    def fund016(self):
        return kurtosis(self.ex_log_ret, 126)

    def fund017(self):
        return kurtosis(self.ex_log_ret, 252)

    def fund018(self):
        return kurtosis(self.ex_log_ret, 252 * 2)

    def fund019(self):
        return kurtosis(self.ex_log_ret, 252 * 3)

    def fund020(self):
        return kurtosis(self.ex_log_ret, 252 * 5)

    # fund021-025 半年、年、2年、3年、5年对数收益的偏度
    def fund021(self):
        return skewness(self.log_ret, 126)

    def fund022(self):
        return skewness(self.log_ret, 252)

    def fund023(self):
        return skewness(self.log_ret, 252 * 2)

    def fund024(self):
        return skewness(self.log_ret, 252 * 3)

    def fund025(self):
        return skewness(self.log_ret, 252 * 5)

    # fund026-030 半年、年、2年、3年、5年超额对数收益的偏度
    def fund026(self):
        return skewness(self.ex_log_ret, 126)

    def fund027(self):
        return skewness(self.ex_log_ret, 252)

    def fund028(self):
        return skewness(self.ex_log_ret, 252 * 2)

    def fund029(self):
        return skewness(self.ex_log_ret, 252 * 3)

    def fund030(self):
        return skewness(self.ex_log_ret, 252 * 5)

    # fund031-035 半年、年、2年、3年、5年对数收益的胜率
    def fund031(self):
        return positive_ratio(self.log_ret, 126)

    def fund032(self):
        return positive_ratio(self.log_ret, 252)

    def fund033(self):
        return positive_ratio(self.log_ret, 252 * 2)

    def fund034(self):
        return positive_ratio(self.log_ret, 252 * 3)

    def fund035(self):
        return positive_ratio(self.log_ret, 252 * 5)

    # fund036-040 半年、年、2年、3年、5年超额对数收益的胜率
    def fund036(self):
        return positive_ratio(self.ex_log_ret, 126)

    def fund037(self):
        return positive_ratio(self.ex_log_ret, 252)

    def fund038(self):
        return positive_ratio(self.ex_log_ret, 252 * 2)

    def fund039(self):
        return positive_ratio(self.ex_log_ret, 252 * 3)

    def fund040(self):
        return positive_ratio(self.ex_log_ret, 252 * 5)

    # fund041-045 半年、年、2年、3年、5年对数收益的年化波动率
    def fund041(self):
        return ts_volatility(self.log_ret, 126)

    def fund042(self):
        return ts_volatility(self.log_ret, 252)

    def fund043(self):
        return ts_volatility(self.log_ret, 252 * 2)

    def fund044(self):
        return ts_volatility(self.log_ret, 252 * 3)

    def fund045(self):
        return ts_volatility(self.log_ret, 252 * 5)

    # fund046-050 半年、年、2年、3年、5年超额对数收益的年化波动率
    def fund046(self):
        return ts_volatility(self.ex_log_ret, 126)

    def fund047(self):
        return ts_volatility(self.ex_log_ret, 252)

    def fund048(self):
        return ts_volatility(self.ex_log_ret, 252 * 2)

    def fund049(self):
        return ts_volatility(self.ex_log_ret, 252 * 3)

    def fund050(self):
        return ts_volatility(self.ex_log_ret, 252 * 5)

    # fund051-055计算半年、年、2年、3年、5年的最大回撤
    def fund051(self):
        return self.close.rolling(126).apply(rolling_maxretrace)

    def fund052(self):
        return self.close.rolling(252).apply(rolling_maxretrace)

    def fund053(self):
        return self.close.rolling(252 * 2).apply(rolling_maxretrace)

    def fund054(self):
        return self.close.rolling(252 * 3).apply(rolling_maxretrace)

    def fund055(self):
        return self.close.rolling(252 * 5).apply(rolling_maxretrace)

    # fund056-060计算半年、年、2年、3年、5年的高水位回撤率
    def fund056(self):
        return self.close.rolling(126).apply(rolling_avgretrace)

    def fund057(self):
        return self.close.rolling(252).apply(rolling_avgretrace)

    def fund058(self):
        return self.close.rolling(252 * 2).apply(rolling_avgretrace)

    def fund059(self):
        return self.close.rolling(252 * 3).apply(rolling_avgretrace)

    def fund060(self):
        return self.close.rolling(252 * 5).apply(rolling_avgretrace)

    # fund061-065计算半年、年、2年、3年、5年收益为负的年化标准差
    def fund061(self):
        target_return = log(pow(1.03, 1 / 252))
        asset_return = pd.Series(self.log_ret)
        downside_return = asset_return - target_return
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(126).std() * np.sqrt(252)

    def fund062(self):
        target_return = log(pow(1.03, 1 / 252))
        asset_return = pd.Series(self.log_ret)
        downside_return = asset_return - target_return
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252).std() * np.sqrt(252)

    def fund063(self):
        target_return = log(pow(1.03, 1 / 252))
        asset_return = pd.Series(self.log_ret)
        downside_return = asset_return - target_return
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 2).std() * np.sqrt(252)

    def fund064(self):
        target_return = log(pow(1.03, 1 / 252))
        asset_return = pd.Series(self.log_ret)
        downside_return = asset_return - target_return
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 3).std() * np.sqrt(252)

    def fund065(self):
        target_return = log(pow(1.03, 1 / 252))
        asset_return = pd.Series(self.log_ret)
        downside_return = asset_return - target_return
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 5).std() * np.sqrt(252)

    # fund066-070计算半年、年、2年、3年、5年收益为负的超额年化标准差
    def fund066(self):
        downside_return = pd.Series(self.ex_log_ret)
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(126).std() * np.sqrt(252)

    def fund067(self):
        downside_return = pd.Series(self.ex_log_ret)
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252).std() * np.sqrt(252)

    def fund068(self):
        downside_return = pd.Series(self.ex_log_ret)
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 2).std() * np.sqrt(252)

    def fund069(self):
        downside_return = pd.Series(self.ex_log_ret)
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 3).std() * np.sqrt(252)

    def fund070(self):
        downside_return = pd.Series(self.ex_log_ret)
        downside_return = downside_return.fillna(value=0)
        downside_return[downside_return > 0] = 0
        return downside_return.rolling(252 * 5).std() * np.sqrt(252)

    # fund071-075计算半年、年、2年、3年、5年的夏普比率
    def fund071(self):
        return self.log_ret.rolling(126).apply(yearsharpRatio)

    def fund072(self):
        return self.log_ret.rolling(252).apply(yearsharpRatio)

    def fund073(self):
        return self.log_ret.rolling(252 * 2).apply(yearsharpRatio)

    def fund074(self):
        return self.log_ret.rolling(252 * 3).apply(yearsharpRatio)

    def fund075(self):
        return self.log_ret.rolling(252 * 5).apply(yearsharpRatio)

    # fund076-080计算半年、年、2年、3年、5年的索提诺比率
    def fund076(self):
        return self.log_ret.rolling(126).apply(sortinoratio)

    def fund077(self):
        return self.log_ret.rolling(252).apply(sortinoratio)

    def fund078(self):
        return self.log_ret.rolling(252 * 2).apply(sortinoratio)

    def fund079(self):
        return self.log_ret.rolling(252 * 3).apply(sortinoratio)

    def fund080(self):
        return self.log_ret.rolling(252 * 5).apply(sortinoratio)

    # fund081-085计算半年、年、2年、3年、5年的alpha
    def fund081(self):
        windows = 126
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[0]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund082(self):
        windows = 252
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[0]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund083(self):
        windows = 252 * 2
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[0]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund084(self):
        windows = 252 * 3
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[0]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund085(self):
        windows = 252 * 5
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[0]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    # fund086-090计算半年、年、2年、3年、5年的beta
    def fund086(self):
        windows = 126
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[1]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund087(self):
        windows = 252
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[1]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund088(self):
        windows = 252 * 2
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[1]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund089(self):
        windows = 252 * 3
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[1]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    def fund090(self):
        windows = 252 * 5
        rf = log(pow(1.03, 1 / 252))
        y = self.log_ret - rf
        x = self.ex_log_ret - rf
        solution = PandasRollingOLS(y, x, windows).solution
        alpha = list(zip(*solution))[1]
        alpha0 = [None] * (windows - 1)
        alpha0.extend(alpha)
        return alpha0

    # fund091-095计算半年、年、2年、3年、5年的var值
    def fund091(self):
        return self.log_ret.rolling(126).apply(rolling_var)

    def fund092(self):
        return self.log_ret.rolling(252).apply(rolling_var)

    def fund093(self):
        return self.log_ret.rolling(252 * 2).apply(rolling_var)

    def fund094(self):
        return self.log_ret.rolling(252 * 3).apply(rolling_var)

    def fund095(self):
        return self.log_ret.rolling(252 * 5).apply(rolling_var)

    # fund096-100计算半年、年、2年、3年、5年的ir值
    def fund096(self):
        windows = 126
        asset_annualized_return = self.log_ret.rolling(windows).mean()
        index_annualized_return = self.log_benchmark.rolling(windows).mean()
        tracking_error = self.ex_log_ret.rolling(windows).std(ddof=1)
        return (asset_annualized_return - index_annualized_return) / tracking_error * np.sqrt(252)

    def fund097(self):
        windows = 252
        asset_annualized_return = self.log_ret.rolling(windows).mean()
        index_annualized_return = self.log_benchmark.rolling(windows).mean()
        tracking_error = self.ex_log_ret.rolling(windows).std(ddof=1)
        return (asset_annualized_return - index_annualized_return) / tracking_error * np.sqrt(252)

    def fund098(self):
        windows = 252 * 2
        asset_annualized_return = self.log_ret.rolling(windows).mean()
        index_annualized_return = self.log_benchmark.rolling(windows).mean()
        tracking_error = self.ex_log_ret.rolling(windows).std(ddof=1)
        return (asset_annualized_return - index_annualized_return) / tracking_error * np.sqrt(252)

    def fund099(self):
        windows = 252 * 3
        asset_annualized_return = self.log_ret.rolling(windows).mean()
        index_annualized_return = self.log_benchmark.rolling(windows).mean()
        tracking_error = self.ex_log_ret.rolling(windows).std(ddof=1)
        return (asset_annualized_return - index_annualized_return) / tracking_error * np.sqrt(252)

    def fund100(self):
        windows = 252 * 5
        asset_annualized_return = self.log_ret.rolling(windows).mean()
        index_annualized_return = self.log_benchmark.rolling(windows).mean()
        tracking_error = self.ex_log_ret.rolling(windows).std(ddof=1)
        return (asset_annualized_return - index_annualized_return) / tracking_error * np.sqrt(252)

    # fund101-105计算半年、年、2年、3年、5年的收益回撤比
    def fund101(self):
        return self.fund001() / self.fund051()

    def fund102(self):
        return self.fund002() / self.fund052()

    def fund103(self):
        return self.fund003() / self.fund053()

    def fund104(self):
        return self.fund004() / self.fund054()

    def fund105(self):
        return self.fund005() / self.fund055()

    def alpha801(self):
        '''
        :return: momentum4,8
        '''
        SK = 4
        LK = 8
        price_return = self.close / self.close.shift(1) - 1
        volAdjRet = price_return / price_return.ewm(span=SK, min_periods=SK, adjust=False).std(bias=True)
        px_df = np.cumsum(volAdjRet)
        sig = px_df.ewm(span=SK, min_periods=SK).mean() - px_df.ewm(span=LK, min_periods=SK).mean()
        sig_normalized = sig / vol_estimator_garch(sig, 25)
        sig_resp = get_resp_curve(sig_normalized, 'gaussian')
        os_norm = 1.0 / 0.89
        sig = sig_normalized * sig_resp * os_norm
        return sig

    def alpha802(self):
        '''
        :return: momentum8,16
        '''
        SK = 8
        LK = 16
        price_return = self.close / self.close.shift(1) - 1
        volAdjRet = price_return / price_return.ewm(span=SK, min_periods=SK, adjust=False).std(bias=True)
        px_df = np.cumsum(volAdjRet)
        sig = px_df.ewm(span=SK, min_periods=SK).mean() - px_df.ewm(span=LK, min_periods=SK).mean()
        sig_normalized = sig / vol_estimator_garch(sig, 25)
        sig_resp = get_resp_curve(sig_normalized, 'gaussian')
        os_norm = 1.0 / 0.89
        sig = sig_normalized * sig_resp * os_norm
        return sig

    def alpha803(self):
        '''
        :return: momentum8,16
        '''
        SK = 16
        LK = 32
        price_return = self.close / self.close.shift(1) - 1
        volAdjRet = price_return / price_return.ewm(span=SK, min_periods=SK, adjust=False).std(bias=True)
        px_df = np.cumsum(volAdjRet)
        sig = px_df.ewm(span=SK, min_periods=SK).mean() - px_df.ewm(span=LK, min_periods=SK).mean()
        sig_normalized = sig / vol_estimator_garch(sig, 25)
        sig_resp = get_resp_curve(sig_normalized, 'gaussian')
        os_norm = 1.0 / 0.89
        sig = sig_normalized * sig_resp * os_norm
        return sig

    def alpha804(self):
        '''
        :return: momentum8,16
        '''
        SK = 32
        LK = 64
        price_return = self.close / self.close.shift(1) - 1
        volAdjRet = price_return / price_return.ewm(span=SK, min_periods=SK, adjust=False).std(bias=True)
        px_df = np.cumsum(volAdjRet)
        sig = px_df.ewm(span=SK, min_periods=SK).mean() - px_df.ewm(span=LK, min_periods=SK).mean()
        sig_normalized = sig / vol_estimator_garch(sig, 25)
        sig_resp = get_resp_curve(sig_normalized, 'gaussian')
        os_norm = 1.0 / 0.89
        sig = sig_normalized * sig_resp * os_norm
        return sig

    def alpha805(self):
        '''
        :return: momentum8,16
        '''
        SK = 64
        LK = 128
        price_return = self.close / self.close.shift(1) - 1
        volAdjRet = price_return / price_return.ewm(span=SK, min_periods=SK, adjust=False).std(bias=True)
        px_df = np.cumsum(volAdjRet)
        sig = px_df.ewm(span=SK, min_periods=SK).mean() - px_df.ewm(span=LK, min_periods=SK).mean()
        sig_normalized = sig / vol_estimator_garch(sig, 25)
        sig_resp = get_resp_curve(sig_normalized, 'gaussian')
        os_norm = 1.0 / 0.89
        sig = sig_normalized * sig_resp * os_norm
        return sig

    def alpha001(self):
        data_x = rank(delta(log(self.volume), 1))
        data_y = rank(((self.close - self.open) / self.open))
        data = correlation(data_x, data_y, 6) * -1
        return data

    def alpha002(self):
        data_m = delta((((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)), 1)
        data_m = data_m * -1
        return data_m

    def alpha003(self):
        data_mid1 = min_s(self.low, delay(self.close, 1))
        data_mid2 = max_s(self.high, delay(self.close, 1))
        data_mid3 = [z if x > y else v for x, y, z, v in zip(self.close, delay(self.close, 1), data_mid1, data_mid2)]
        data_mid3 = np.array(data_mid3)
        data_mid4 = self.close - data_mid3
        data_mid5 = [0 if x == y else z for x, y, z in zip(self.close, delay(self.close, 1), data_mid4)]
        data_mid5 = np.array(data_mid5)
        df = pd.Series(data_mid5, name="value")
        a = ts_sum(df, 6)
        return a

    def alpha004(self):
        data_mid1 = self.volume / (sma(self.volume, 20))
        data_mid2 = [1 if x >= 1 else -1 for x in data_mid1]
        data_mid3 = [1 if x < y else z for x, y, z in
                     zip((ts_sum(self.close, 2) / 2), ((ts_sum(self.close, 8) / 8) - (stddev(self.close, 8))),
                         data_mid2)]
        data_mid4 = [-1 if x < y else z for x, y, z in
                     zip((ts_sum(self.close, 8) / 8 + stddev(self.close, 8)), (ts_sum(self.close, 2) / 2), (data_mid3))]
        return pd.Series(data_mid4, name="value")

    def alpha005(self):
        data_mid1 = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        return -1 * ts_max(data_mid1)

    def alpha006(self):
        return -1 * (rank(sign(delta((self.open * 0.85 + self.high * 0.15), 4))))

    def alpha009(self):
        data_mid1 = ((self.high + self.low) / 2 - (delay(self.high) + delay(self.low)) / 2) * (
                self.high - self.low) / self.volume
        return sma_list(data_mid1, 7, 2)

    def alpha011(self):
        data_mid1 = ((self.close - self.low) - (self.high - self.low)) / (self.high - self.low) * self.volume
        return ts_sum(data_mid1, 6)

    def alpha014(self):
        return self.close - delay(self.close, 5)

    def alpha015(self):
        return self.open / delay(self.close) - 1

    def alpha018(self):
        return self.close / delay(self.close, 5)

    def alpha019(self):
        data_mid1 = [0 if x == y else z for x, y, z in
                     zip((self.close), (delay(self.close, 5)), (self.close - delay(self.close, 5)) / self.close)]
        data_mid2 = [z if x < y else v for x, y, z, v in
                     zip(self.close, delay(self.close, 5), (self.close - delay(self.close, 5)) / delay(self.close, 5),
                         data_mid1)]
        return pd.Series(data_mid2, name="value")

    def alpha020(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100

    def alpha024(self):
        data_mid = self.close - delay(self.close, 5)
        return sma_list(data_mid, 5, 1)

    def alpha028(self):
        data_mid1 = (self.close - ts_min(self.low, 9) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100)
        data_mid1 = sma_list(data_mid1, 3, 1)
        data_mid2 = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_max(self.low, 9)) * 100
        data_mid2 = sma_list(data_mid2, 3, 1)
        data_mid3 = sma_list(data_mid2, 3, 1)
        return 3 * data_mid1 - 2 * data_mid3

    def alpha029(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume

    def alpha031(self):
        return (self.close - sma(self.close, 12)) / sma(self.close, 12) * 100

    def alpha032(self):
        return -1 * ts_sum((rank(correlation(rank(self.high), rank(self.volume), 3))), 3)

    def alpha033(self):
        data_mid1 = -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)
        data_mid2 = rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        return data_mid1 * data_mid2 * ts_rank(self.volume, 5)

    def alpha034(self):
        return sma(self.close, 12) / self.close

    def alpha035(self):
        data_mid1 = rank(decay_linear(delta(self.open), 15))
        data_mid2 = rank(decay_linear(correlation(self.volume, self.open, 17), 7))
        return min_s(data_mid1, data_mid2) * -1

    def alpha037(self):
        data_mid1 = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        data_mid2 = delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)
        return rank(data_mid1 - data_mid2) * -1

    def alpha038(self):
        data = [z if x < y else 0 for x, y, z in zip(ts_sum(self.high, 20) / 20, self.high, (-1 * delta(self.high, 2)))]
        return pd.Series(data, name="value")

    def alpha040(self):
        data_mid1 = copy.deepcopy(self.volume)
        data_mid1 = [0.001 if x <= y else z for x, y, z in zip(self.close, delay(self.close), data_mid1)]
        data_mid1 = pd.Series(data_mid1, name="value")
        data_mid2 = copy.deepcopy(self.volume)
        data_mid2 = [0.001 if x > y else z for x, y, z in zip(self.close, delay(self.close), data_mid2)]
        data_mid2 = pd.Series(data_mid2, name="value")
        return ts_sum(data_mid1, 26) / ts_sum(data_mid2, 26)

    def alpha042(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha043(self):
        data_mid1 = -1 * copy.deepcopy(self.volume)
        data_mid1[self.close >= delay(self.close)] = 0
        data_mid2 = copy.deepcopy(self.volume)
        data_mid2[self.close <= delay(self.close)] = data_mid1
        return ts_sum(data_mid2, 6)

    def alpha046(self, n=3):
        data_mid1 = sma(self.close, n) + sma(self.close, n * 2) + sma(self.close, n * 4) + sma(self.close, n * 8)
        return -data_mid1 / (4 * self.close)

    def alpha047(self, n=3):
        data_mid1 = (ts_max(self.high, n * 2) - self.close)
        data_mid2 = ts_max(self.high, n * 2) - ts_min(self.low, n * 2)
        data_mid3 = sma_list(data_mid1 / data_mid2, n * 3, 1)
        return -100 * data_mid3

    def alpha049(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid1 = ts_sum(data_mid1, 12)
        data_mid2 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)
        return data_mid2 / (data_mid1 + data_mid2)

    def alpha050(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid1 = ts_sum(data_mid1, 12)

        data_mid2 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)

        data_mid3 = (data_mid2 - data_mid1) / (data_mid1 + data_mid2)

        return data_mid3

    def alpha051(self):
        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)
        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)
        data_mid6 = data_mid4 / (data_mid4 + data_mid5)
        return data_mid6

    def alpha052(self):
        data_mid1 = self.high - delay((self.high + self.low + self.close) / 3)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = delay((self.high + self.low + self.close) / 3) - self.low
        data_mid2[data_mid2 < 0] = 0
        return ts_sum(data_mid1, 26) / ts_sum(data_mid2, 26) * 100

    def alpha053(self):
        data_mid1 = ts_count(delay(self.close), self.close, 12)
        return (data_mid1 / 12) * 100

    def alpha055(self):
        data_mid1 = (self.close - delay(self.close) + (self.close - self.open) / 2 + delay(self.close) - delay(
            self.open)) * 16

        data_mid_z = (self.high - delay(self.close)).abs() + (self.low - delay(self.close)).abs() / 2 + (
                delay(self.close) - delay(self.open)).abs() / 4
        data_mid_vz = (self.low - delay(self.close)).abs() + (self.high - delay(self.close)).abs() / 2 + (
                delay(self.close) - delay(self.open)).abs() / 4
        data_mid_vv = (self.high - delay(self.low)).abs() + (delay(self.close) - delay(self.open)) / 4

        data_mid_v = [vz if x1 > y1 and x2 > y2 else vv for x1, y1, x2, y2, vz, vv in
                      zip((self.low - delay(self.close)).abs(), (self.high - delay(self.low)).abs(),
                          (self.low - delay(self.close)).abs(), (self.high - delay(self.close)).abs(), data_mid_vz,
                          data_mid_vv)]
        data_mid2 = [z if x1 > y1 and x2 > y2 else v for x1, y1, x2, y2, z, v in
                     zip((self.high - delay(self.close)).abs(), (self.low - delay(self.close)).abs(),
                         (self.high - delay(self.close)).abs(), (self.high - delay(self.low)).abs(), data_mid_z,
                         data_mid_v)]

        data_mid3 = max_s((self.high - delay(self.close)).abs(), (self.low - delay(self.close)).abs())

        data_all = data_mid1 / data_mid2 * data_mid3

        return ts_sum(data_all, 20)

    def alpha057(self):
        data_mid1 = (self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100
        return sma_list(data_mid1, 3, 1)

    def alpha058(self):
        data_mid1 = ts_count(delay(self.close), self.close, 20)
        return (data_mid1 / 20) * 100

    def alpha059(self):
        data_mid1 = [z if x > y else v for x, y, z, v in
                     zip(self.close, delay(self.close), min_s(self.low, delay(self.close)),
                         max_s(self.high, delay(self.close)))]
        data_mid1 = np.array(data_mid1)
        data_mid1 = self.close.values - data_mid1
        data_mid2 = [0 if x == y else z for x, y, z in zip(self.close, delay(self.close), data_mid1)]
        data_mid2 = pd.Series(data_mid2, name="values")
        return ts_sum(data_mid2, 20)

    def alpha060(self):
        data_mid1 = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        return ts_sum(data_mid1, 20)

    def alpha063(self):
        data_mid1 = [0 if x <= 0 else x for x in (self.close - delay(self.close))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid2 = (self.close - delay(self.close)).abs()
        return ((sma_list(data_mid1, 6, 1)) / (sma_list(data_mid2, 6, 1))) * 100

    def alpha065(self):
        return self.close / sma(self.close, 6)

    def alpha066(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100

    def alpha067(self):
        data_mid1 = [0 if x <= 0 else x for x in (self.close - delay(self.close))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid2 = (self.close - delay(self.close)).abs()
        return ((sma_list(data_mid1, 24, 1)) / (sma_list(data_mid2, 24, 1))) * 100

    def alpha068(self):
        data_mid1 = ((self.high + self.low) / 2 - (delay(self.high) + delay(self.low)) / 2) * (
                self.high - self.low) / self.volume
        return sma_list(data_mid1, 15, 2)

    def alpha069(self):
        dtm = [0 if x <= y else z for x, y, z in
               zip(self.open, delay(self.open), max_s((self.high - self.open), (self.open - delay(self.open))))]
        dbm = [0 if x >= y else z for x, y, z in
               zip(self.open, delay(self.open), max_s((self.open - self.low), (self.open - delay(self.open))))]
        dtm = pd.Series(dtm, name="dtm")
        dbm = pd.Series(dbm, name="dbm")
        data_mid_z = (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dtm, 20)
        data_mid_vz = (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dbm, 20)

        data_mid_v = [0 if x == y else z for x, y, z in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_vz)]
        data_mid = [z if x > y else v for x, y, z, v in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_z, data_mid_v)]

        return pd.Series(data_mid, name="values")

    def alpha070(self):
        return stddev(self.amount, 6)

    def alpha071(self):
        return (self.close - sma(self.close, 24)) / sma(self.close, 24) * 100

    def alpha072(self):
        data_mid1 = (ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100
        return -sma_list(data_mid1, 15, 1)

    def alpha076(self):
        data_mid1 = stddev((self.close / delay(self.close) - 1).abs() / self.volume, 20)
        data_mid2 = sma((self.close / delay(self.close) - 1).abs() / self.volume, 20)
        return data_mid1 / data_mid2

    def alpha078(self):
        data_mid1 = (self.high + self.low + self.close) / 3 + sma((self.high + self.low + self.close) / 3, 12)
        data_mid2 = 0.015 * sma((self.close - sma((self.high + self.low + self.close) / 3, 12)).abs(), 12)

        return data_mid1 / data_mid2

    def alpha079(self):
        data_mid1 = self.close - delay(self.close)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = (self.close - delay(self.close)).abs()
        return (sma_list(data_mid1, 12, 1)) / (sma_list(data_mid2, 12, 1)) * 100

    def alpha080(self):
        return (self.volume - delay(self.volume, 5)) / delay(self.volume, 5) * 100

    def alpha081(self):
        return sma_list(self.volume, 21, 2)

    def alpha082(self):
        data_mid1 = ts_max(self.high, 6) - self.close
        data_mid2 = ts_max(self.high, 6) - ts_min(self.low, 6)
        return -sma_list(data_mid1 / data_mid2 * 100, 20, 1)

    def alpha084(self):
        data_mid_v = [-z if x < y else 0 for x, y, z in zip(self.close, delay(self.close), self.volume)]
        data_mid2 = [z if x > y else v for x, y, z, v in zip(self.close, delay(self.close), self.volume, data_mid_v)]
        data_mid2 = pd.Series(data_mid2, name="values")
        return ts_sum(data_mid2, 20)

    def alpha085(self):
        data_mid1 = ts_rank((self.volume / sma(self.volume, 20)), 20)
        data_mid2 = ts_rank((-1 * delta(self.close, 7)), 8)
        return data_mid1 * data_mid2

    def alpha086(self):
        data_yx = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        data_y = [1 if x < 0 else y for x, y in zip(data_yx, -1 * (self.close - delay(self.close)))]
        data = [-1 if x > 0.25 else y for x, y in zip(data_yx, data_y)]
        data = pd.Series(data, name="values")
        return data

    def alpha088(self):
        return (self.close - delay(self.close, 20)) / delay(self.close, 20) * 100

    def alpha089(self):
        data_mid1 = sma_list(self.close, 13, 2) - sma_list(self.close, 27, 2) - sma_list(
            (sma_list(self.close, 13, 2) - sma_list(self.close, 27, 2)), 10, 2)
        return data_mid1 * 2

    def alpha093(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip(self.open, delay(self.open), max_s((self.open - self.low), (
                self.open - delay(self.open))) / delay(self.open))]
        data_mid1 = pd.Series(data_mid1, name="values")
        return -ts_sum(data_mid1, 20)

    def alpha094(self):
        data_mid_v = [-z if x < y else 0 for x, y, z in zip(self.close, delay(self.close), self.volume)]
        data = [z if x > y else v for x, y, z, v in zip(self.close, delay(self.close), self.volume, data_mid_v)]
        data = pd.Series(data, name="values")
        return ts_sum(data, 30) / self.volume

    def alpha095(self):
        return stddev(self.amount, 20)

    def alpha096(self):
        return sma_list(
            (sma_list(((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100), 3, 1)),
            3, 1)

    def alpha097(self):
        return stddev(self.volume, 10)

    def alpha098(self):
        data_mid = [y if x <= 0.05 else z for x, y, z in
                    zip((delta((ts_sum(self.close, 100) / 100), 100) / delay((self.close), 100)),
                        (-1 * (self.close - ts_min(self.close, 100))), (-1 * (delta(self.close, 3))))]
        return pd.Series(data_mid, name="values")

    def alpha100(self):
        return stddev(self.volume, 20)

    def alpha102(self):
        data_mid = self.volume - delay(self.volume)
        data_mid[data_mid < 0] = 0
        data_mid2 = (self.volume - delay(self.volume)).abs()
        return (sma_list(data_mid, 6, 1)) / (sma_list(data_mid2, 6, 1)) * 100

    def alpha103(self):
        return ((20 - ts_lowday(self.low, 20)) / 20) * 100

    def alpha106(self):
        return self.close - delay(self.close, 20)

    def alpha109(self):
        data_mid1 = sma_list(self.high - self.low, 10, 2)
        return data_mid1 / sma_list(data_mid1, 10, 2)

    def alpha110(self):
        data_mid1 = self.high - delay(self.close)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = delay(self.close) - self.low
        data_mid2[data_mid2 < 0] = 0
        return (ts_sum(data_mid1, 20)) / (ts_sum(data_mid2, 20)) * 100

    def alpha111(self):
        data_mid1 = ((2 * self.close - self.low - self.high) / (self.high - self.low)) * self.volume
        return sma_list(data_mid1, 11, 2) - sma_list(data_mid1, 4, 2)

    def alpha112(self):
        data_mid1 = self.close - delay(self.close)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = self.close - delay(self.close)
        data_mid2[data_mid2 > 0] = 0
        data_mid2 = data_mid2.abs()
        return (ts_sum((data_mid1), 12) - ts_sum((data_mid2), 12)) / (
                ts_sum(data_mid1, 12) + ts_sum(data_mid2, 12)) * 100

    def alpha117(self):
        return (ts_rank(self.volume, 32) * (1 - ts_rank((self.close + self.high - self.low), 16))) * (
                1 - ts_rank(self.returns, 32))

    def alpha118(self):
        return ts_sum((self.high - self.open), 20) / ts_sum((self.open - self.low), 20) * 100

    def alpha122(self):
        data_mid1 = sma_list((sma_list((sma_list((self.close.map(np.log)), 13, 2)), 13, 2)), 13, 2)
        return (data_mid1 - delay(data_mid1)) / (delay(data_mid1))

    def alpha126(self):
        return (self.close + self.high + self.low) / 3

    def alpha128(self):
        data_mid1 = (self.high + self.low + self.close) / 3 * self.volume
        data_mid1[(self.high + self.low + self.close) / 3 <= delay((self.high + self.low + self.close) / 3)] = 0
        data_mid2 = (self.high + self.low + self.close) / 3 * self.volume
        data_mid2[(self.high + self.low + self.close) / 3 >= delay((self.high + self.low + self.close) / 3)] = 0
        return 100 - (100 / (1 + ts_sum((data_mid1), 14) / ts_sum((data_mid2), 14)))

    def alpha129(self):
        data_mid1 = (self.close - delay(self.close)).abs() / delay(self.close)
        data_mid1[self.close >= delay(self.close)] = 0
        return -ts_sum(data_mid1, 12)

    def alpha132(self):
        return sma(self.amount, 20)

    def alpha133(self):
        return ((20 - ts_highday(self.high, 20)) / 20) * 100 - ((20 - ts_lowday(self.low, 20)) / 20) * 100

    def alpha134(self):
        return (self.close - delay(self.close, 12)) / delay(self.close, 12)

    def alpha135(self, m=4):
        data_mid1 = delay(self.close / delay(self.close, 5 * m))
        return sma_list(data_mid1, 5 * m, 1)

    def alpha137(self):
        data_mid1 = (self.high - delay(self.low)).abs() - (delay(self.close) - delay(self.open)).abs() / 4
        data_mid1[((self.low - delay(self.close)).abs() > (self.high - delay(self.low)).abs()) & (
                (self.low - delay(self.close)).abs() > (self.high - delay(self.close)).abs())] = (self.low - delay(
            self.close)).abs() + (self.high - delay(self.close)).abs() / 2 + (delay(self.close) - delay(
            self.open)).abs() / 4
        data_mid1[((self.high - delay(self.close)).abs() > (self.low - delay(self.close)).abs()) & (
                (self.high - delay(self.close)).abs() > (self.high - delay(self.low)).abs())] = (self.high - delay(
            self.close)).abs() + (self.low - delay(self.close)).abs() / 2 + (delay(self.close) - delay(
            self.open)).abs() / 4

        return 16 * (self.close - delay(self.close) + (self.close - self.open) / 2 + delay(self.close) - delay(
            self.open)) / data_mid1 * max_s((self.high - delay(self.close)).abs(), (self.low - delay(self.close)).abs())

    def alpha139(self):
        return -1 * correlation(self.open, self.volume, 10)

    def alpha145(self):
        return (sma(self.volume, 9) - sma(self.volume, 26)) / sma(self.volume, 12) * 100

    def alpha146(self):
        data_mid1 = sma(((self.close - delay(self.close)) / delay(self.close) - sma_list(
            ((self.close - delay(self.close)) / delay(self.close)), 61, 2)), 20)
        data_mid2 = (self.close - delay(self.close)) / delay(self.close) - sma_list(
            ((self.close - delay(self.close)) / delay(self.close)), 61, 2)
        data_mid3 = (((self.close - delay(self.close)) / delay(self.close) - (
                (self.close - delay(self.close)) / delay(self.close) - (
                (self.close - delay(self.close)) / delay(self.close) - sma_list(
            ((self.close - delay(self.close)) / delay(self.close)), 61, 2)))) ** 2)
        data_mid3 = sma_list(data_mid3, 61, 2)
        return data_mid1 * data_mid2 / data_mid3

    def alpha150(self):
        return (self.close + self.high + self.low) / (3 * self.close)

    def alpha151(self):
        return sma_list((self.close - delay(self.close, 20)), 20, 1)

    def alpha152(self):
        data_mid1 = sma((delay(sma_list((delay(self.close / delay(self.close, 9))), 9, 1))), 12)
        data_mid2 = sma((delay(sma_list((delay(self.close / delay(self.close, 9))), 9, 1))), 26)
        return sma_list(data_mid1 - data_mid2, 9, 1)

    def alpha153(self):
        return -(sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / 4

    def alpha155(self):
        return sma_list(self.volume, 13, 2) - sma_list(self.volume, 27, 2) - sma_list(
            (sma_list(self.volume, 13, 2) - sma_list(self.volume, 27, 2)), 10, 2)

    def alpha158(self):
        return ((self.high - sma_list(self.close, 15, 2)) - (self.low - sma_list(self.close, 15, 2))) / self.close

    def alpha159(self):
        data_mid1 = (self.close - ts_sum((min_s(self.low, delay(self.close))), 6)) / (
            ts_sum(max_s(self.high, delay(self.close)) - min_s(self.low, delay(self.close)), 6)) * 12 * 24
        data_mid2 = (self.close - ts_sum((min_s(self.low, delay(self.close))), 12)) / (
            ts_sum(max_s(self.high, delay(self.close)) - min_s(self.low, delay(self.close)), 12)) * 6 * 24
        data_mid3 = (self.close - ts_sum((min_s(self.low, delay(self.close))), 24)) / (
            ts_sum(max_s(self.high, delay(self.close)) - min_s(self.low, delay(self.close)), 24)) * 6 * 24
        return (data_mid1 + data_mid2 + data_mid3) * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    def alpha160(self):
        data_mid = stddev(self.close, 20)
        data_mid[self.close > delay(self.close)] = 0
        return sma_list(data_mid, 20, 1)

    def alpha161(self):
        return sma(max_s((max_s((self.high - self.low), (delay(self.close) - self.high).abs())),
                         (delay(self.close) - self.low).abs()), 12)

    def alpha162(self):
        data_mid1 = self.close - delay(self.close)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = sma_list(data_mid1, 12, 1) / sma_list((self.close - delay(self.close)), 12, 1) * 100
        data_mid3 = copy.deepcopy(data_mid2)
        data_mid3[data_mid3 > 12] = 12
        data_mid4 = copy.deepcopy(data_mid2)
        data_mid4[data_mid4 < 12] = 12
        return (data_mid2 - data_mid3) / (data_mid4 - data_mid3)

    def alpha164(self):
        data_mid1 = 1 / (self.close - delay(self.close))
        data_mid1[self.close <= delay(self.close)] = 1

        data_mid2 = copy.deepcopy(data_mid1)
        data_mid2[data_mid2 > 12] = 12

        return sma_list(((data_mid1 - data_mid2) / (self.high - self.low) * 100), 13, 2)

    def alpha167(self):
        data_mid = self.close - delay(self.close)
        data_mid[self.close <= delay(self.close)] = 0
        return ts_sum(data_mid, 12) / self.close

    def alpha168(self):
        return -1 * self.volume / sma(self.volume, 20)

    def alpha169(self):
        data_mid = delay(sma_list((self.close - delay(self.close)), 9, 1))
        return sma_list((sma(data_mid, 12) - sma(data_mid, 26)), 10, 1)

    def alpha171(self):
        return (-1 * (self.low - self.close) * (self.open ** 5)) / ((self.close - self.high) * (self.close ** 5))

    def alpha172(self):
        tr = max_s(max_s((self.high - self.low), (self.high - delay(self.close)).abs()),
                   (self.low - delay(self.close)).abs())
        hd = self.high - delay(self.high)
        ld = delay(self.low) - self.low

        data_mid1 = copy.deepcopy(ld)
        data_mid1[(ld <= 0) | (ld <= hd)] = 0
        data_mid1 = 100 * ts_sum(data_mid1, 14) / ts_sum(tr, 14)

        data_mid2 = copy.deepcopy(hd)
        data_mid2[(hd <= 0) | (hd <= ld)] = 0
        data_mid2 = 100 * ts_sum(data_mid2, 14) / ts_sum(tr, 14)

        data_mid3 = (data_mid1 - data_mid2).abs() / (data_mid1 + data_mid2) * 100

        return sma(data_mid3, 6)

    def alpha173(self):
        return sma_list(self.close, 13, 2) * 3 - 2 * sma_list((sma_list(self.close, 13, 2)), 13, 2) + sma_list(
            (sma_list((sma_list((log(self.close)), 13, 2)), 13, 2)), 13, 2)

    def alpha174(self):
        data_mid1 = stddev(self.close, 20)
        data_mid1[self.close <= delay(self.close)] = 0
        return sma_list(data_mid1, 20, 1)

    def alpha175(self):
        return sma(max_s(max_s((self.high - self.low), (delay(self.close) - self.high).abs()),
                         (delay(self.close) - self.low).abs()), 6) / self.close

    def alpha177(self):
        return 100 * ((20 - ts_highday(self.high, 20)) / 20)

    def alpha178(self):
        return self.volume * (self.close - delay(self.close)) / delay(self.close)

    def alpha180(self):
        data_mid1 = (-1 * ts_rank((delta(self.close, 7)).abs(), 60)) * sign(delta(self.close, 7))
        data_mid1[sma(self.volume, 20) >= self.volume] = -1 * self.volume
        return data_mid1

    def alpha186(self):
        tr = max_s(max_s((self.high - self.low), (self.high - delay(self.close)).abs()),
                   (self.low - delay(self.close)).abs())
        hd = self.high - delay(self.high)
        ld = delay(self.low) - self.low

        data_mid1 = copy.deepcopy(ld)
        data_mid1[(ld <= 0) | (ld <= hd)] = 0
        data_mid1 = 100 * ts_sum(data_mid1, 14) / ts_sum(tr, 14)

        data_mid2 = copy.deepcopy(hd)
        data_mid2[(hd <= 0) | (hd <= ld)] = 0
        data_mid2 = 100 * ts_sum(data_mid2, 14) / ts_sum(tr, 14)

        data_mid3 = (data_mid1 - data_mid2).abs() / (data_mid1 + data_mid2) * 100
        data_mid3_sma = sma(data_mid3, 6)

        return (data_mid3_sma + delay(data_mid3_sma, 6)) / 2

    def alpha187(self):
        data_mid1 = max_s((self.high - self.open), (self.open - delay(self.open)))
        # data_mid1[self.open<=delay(self.open)]=0
        data_mid1 = [0 if x <= y else z for x, y, z in zip(self.open, delay(self.open), data_mid1)]
        data_mid1 = pd.Series(data_mid1, name="value")
        return ts_sum(data_mid1, 20) / self.close

    def alpha188(self):
        return ((self.high - self.low - sma_list((self.high - self.low), 11, 2)) / (
            sma_list((self.high - self.low), 11, 2))) * 100

    def alpha189(self):
        return sma((self.close - sma(self.close, 6)) / sma(self.close, 6).abs(), 6)

    def alpha191(self):
        return correlation(sma(self.volume, 20), self.low, 5) + (self.high + self.low) / 2 - self.close

    def alpha192(self):
        data_mid1 = -1 * delta(self.close)
        data_mid1[ts_max(delta(self.close), 5) < 0] = delta(self.close)
        data_mid1[ts_min(delta(self.close), 5) > 0] = delta(self.close)
        return data_mid1

    def alpha193(self):
        return -1 * (correlation(self.open, self.volume))

    def alpha194(self):
        return np.sign(delta(self.volume)) * (-1) * (delta(self.close))

    def alpha195(self):
        data_mid1 = [z if x < y else 0 for x, y, z in
                     zip((ts_sum(self.high, 20) / 20), self.high, (-1 * delta(self.high, 2)))]
        data_mid1 = pd.Series(data_mid1, name="values")
        return data_mid1

    def alpha196(self):
        data_mid1 = -1 * (self.close - delay(self.close))
        data_mid2 = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        data_mid1[data_mid2 < 0] = 1
        data_mid1[data_mid2 > 0.25] = -1
        return data_mid1

    def alpha197(self):
        data_mid1 = -1 * (self.close - delay(self.close))
        data_mid2 = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        data_mid1[data_mid2 < -0.1] = 1
        return data_mid1

    def alpha198(self):
        data_mid1 = -1 * (self.close - delay(self.close))
        data_mid2 = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        data_mid1[data_mid2 < -0.05] = 1
        return data_mid1

    def alpha199(self):
        data_mid1 = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low)
        return -1 * delta(data_mid1, 9)

    def alpha200(self):
        data_mid1 = -1 * (self.low - self.close) * (self.open ** 5)
        data_mid2 = (self.low - self.high) * (self.close ** 5)
        return data_mid1 / data_mid2

    def alpha201(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    def get_alpha_methods(self):
        return list(filter(lambda x: x.startswith('alpha') and callable(getattr(self, x)), dir(self)))

    def get_fund_methods(self):
        return list(filter(lambda x: x.startswith('fund') and callable(getattr(self, x)), dir(self)))


# Alpha=Alphas(data)
# alpha_use=Alpha.alpha162()
# print(alpha_use)
'''
a=list(range(1,192))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))
#print(alpha_test)
#print(alpha_use)
i=0
for func in alpha_test:
    print(func)
    try:
        eval(func)()
        i+=1
        print(i)
    except AttributeError:
        pass
'''
