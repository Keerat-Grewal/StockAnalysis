import numpy as np
from talib import abstract
import plotly.graph_objects as go
import matplotlib.dates as mpl_dates
import yfinance as yf
import pandas as pd
import os

# basic stock class
class Stock:

    def __init__(self, ticker, period, interval):
        self.ticker = yf.Ticker(ticker)
        self.stock_data = self.ticker.history(period=period, interval=interval)
        self.stock_data["Date"] = pd.to_datetime(self.stock_data.index)
        self.stock_data["Date"] = self.stock_data["Date"].apply(mpl_dates.date2num)
        self.opening_prices = self.stock_data["Open"]
        self.closing_prices = self.stock_data["Close"]
        # remove unnecessary columns
        self.stock_data = self.stock_data.drop(['Dividends', 'Stock Splits'], axis=1)


    def get_MA(self, timeperiod):
        MA = abstract.Function('ma')
        res = MA(self.closing_prices, timeperiod=timeperiod)
        # try to add to pandas series frame here
        title = f"MA {timeperiod}"
        self.stock_data[title] = res
        return res


    def get_MACD(self):
        MACD = abstract.Function('macd')
        macd, macdsignal, macdhist = MACD(self.closing_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        self.stock_data["MACD"] = macd
        self.stock_data["MACD signal"] = macdsignal
        self.stock_data["MACD hist"] = macdhist
        return macd, macdsignal, macdhist


    def get_RSI(self):
        RSI = abstract.Function('rsi')
        res = RSI(self.closing_prices, timeperiod=14)
        self.stock_data["RSI"] = res
        return res

