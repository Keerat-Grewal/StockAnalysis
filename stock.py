import numpy as np
from talib import abstract
import plotly.graph_objects as go
import matplotlib.dates as mpl_dates
import yfinance as yf
import pandas as pd
import os


class Stock:

    def __init__(self, ticker, period, interval):
        self.ticker = yf.Ticker(ticker)
        self.stock_data = self.ticker.history(period=period, interval=interval)
        self.stock_data["Date"] = pd.to_datetime(self.stock_data.index)
        self.stock_data["Date"] = self.stock_data["Date"].apply(mpl_dates.date2num)
        #self.stock_data = self.stock_data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
        # self.stock_data['Datetime'] = pd.to_datetime(self.stock_data.index)
        # name = 'SPY'
        # ticker = yf.Ticker(name)
        # df = ticker.history(period=period, interval=interval)
        # print(df)
        self.closing_prices = self.stock_data["Close"]


    def get_MA(self, timeperiod):
        MA = abstract.Function('ma')
        res = MA(self.closing_prices, timeperiod=timeperiod)
        # try to add to pandas series frame here
        return res


    def get_MACD(self):
        MACD = abstract.Function('macd')
        macd, macdsignal, macdhist = MACD(self.closing_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        return macd, macdsignal, macdhist


    def get_RSI(self):
        RSI = abstract.Function('rsi')
        res = RSI(self.closing_prices, timeperiod=14)
        return res
