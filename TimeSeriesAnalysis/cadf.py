import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
# import pandas.io.data as web
import pandas_datareader.data as web
import pprint
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


# from pandas.stats.api import ols


def plot_price_series(df, ts1, ts2):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()


def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()


def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()


if __name__ == "__main__":
    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2020, 1, 1)

    arex = web.get_data_yahoo("AAPL", start, end)
    print(arex)
    wll = web.get_data_yahoo("WLL", start, end)
    print(wll)

    df = pd.DataFrame(index=arex.index)
    df["AREX"] = arex["Adj Close"] / arex["Adj Close"].tolist()[0]

    df["WLL"] = wll["Adj Close"] / wll["Adj Close"].tolist()[0]

    # Plot the two time series
    plot_price_series(df, "AREX", "WLL")

    # Display a scatter plot of the two time series
    plot_scatter_series(df, "AREX", "WLL")

    # Calculate optimal hedge ratio "beta"
    res = sm.OLS(df['WLL'], sm.add_constant(df["AREX"])).fit()
    beta_hr = res.params[1]

    # Calculate the residuals of the linear combination
    df["res"] = df["WLL"] - beta_hr * df["AREX"]

    df_amzn = web.get_data_yahoo("AMZN", datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1))
    df_amzn['res'] = df_amzn["Adj Close"]

    # Plot the residuals
    plot_residuals(df_amzn)

    # Calculate and output the CADF test on the residuals
    cadf = ts.adfuller(df_amzn["res"])
    pprint.pprint(cadf)

