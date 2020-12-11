from stock import *
import numpy as np
import matplotlib.pyplot as plt
#from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import pandas as pd
import math
import os

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers


def main():
    # test with stock
    stock = Stock("FB", "5y", "1d")
    #stock.get_MACD()

    # under 30 means oversold, over 70 means overbought
    RSI = stock.get_RSI()
    #print(f"RSI : {RSI[len(RSI) - 1]}")
    MACD_rating = analyze_MACD(stock)
    #print(MACD_rating)
    check_MA_cross = get_golden_cross(stock)
    #print(check_MA_cross)
    res = final_analysis(RSI[len(RSI) - 1], MACD_rating, check_MA_cross)
    #print(res)


def final_analysis(RSI, MACD_rating, golden_cross):
    print(f'RSI : {RSI}')
    if not golden_cross:
        return "Bad buy, no golden cross present"
    if golden_cross:
        if MACD_rating < 3:
            return "Bad buy, MACD rating < 3"
        if MACD_rating == 4 and golden_cross and (RSI < 30 or RSI < 60):
            return "BEST BUY"
        else:
            return "COULD BE GOOD BUY"


def analyze_MACD(stock):
    MACD = stock.get_MACD()
    actual_MACD = MACD[0]
    #signal_line = MACD[1]
    MACD_hist = MACD[2]
    #print(MACD_hist)

    # best scenario = when MACD crosses above signal line and it is positive  --> 4
    # good scenario = when MACD crosses above signal line and it is negative --> 3
    # bad scenario = when MACD crosses below signal line and it is positive  --> 2
    # worst scenario = when MACD crosses below signal line and it is negative  --> 1

    negative = False
    if MACD_hist[len(MACD_hist) - 1] < 0:
        negative = True

    for i in range(len(MACD_hist) - 2, -1, -1):
        if negative and MACD_hist[i] >= 0:
            if actual_MACD[i] > 0:
                #print(MACD_hist[i])
                return 2
            else:
                #print(MACD_hist[i])
                return 1
        elif not negative and MACD_hist[i] <= 0:
            if actual_MACD[i] > 0:
                #print(MACD_hist[i])
                return 4
            else:
                #print(MACD_hist[i])
                return 3


"""
    ticker : (MA 50 : List of MA, MA 200 : List of MA) -> Boolean
    
    Takes in a Stock and returns whether it has a golden cross
"""
def get_golden_cross(stock):
    times = []
    counter = 1
    golden_cross = False
    moving_average_50 = stock.get_MA(50)
    original_length = len(moving_average_50)
    moving_average_200 = stock.get_MA(200)

    for i in range(len(moving_average_200) - 1, -1, -1):
        if math.isnan(moving_average_200[i]):
            break
        times.append(counter)
        counter += 1

    moving_average_50 = moving_average_50[len(moving_average_50) - len(times):]
    moving_average_200 = moving_average_200[len(moving_average_200) - len(times):]

    if len(moving_average_50) == 0 or len(moving_average_200) == 0:
        raise Exception

    #fig, ax = plt.subplots()

    x_axis = stock.stock_data.index[original_length - len(times):]
    # ax.plot(x_axis, moving_average_50, label="50 MA")
    # ax.plot(x_axis, moving_average_200, label="200 MA")
    # candlestick_ohlc(ax, stock.stock_data.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    #
    # ax.legend(loc='upper right', borderaxespad=0.)
    #
    # date_format = mpl_dates.DateFormatter('%d %b %Y')
    # ax.xaxis.set_major_formatter(date_format)
    # fig.autofmt_xdate()


    intersection_points = []

    if moving_average_50[0] > moving_average_200[0]:
        start = 0
    else:
        start = 1

    for i in range(1, len(moving_average_50)):
        # if the moving_average_50 > moving_average_200
        if start == 0:
            # if moving_average_50 < moving_average_200 then intersection has occurred
            if moving_average_50[i] < moving_average_200[i]:
                # this is golden cross
                if moving_average_50[len(moving_average_200)-1] > moving_average_50[i]:
                    golden_cross = True
                intersection_points.append(x_axis[i])
                start = 1
        # if the moving_average_200 > moving_average_50
        elif start == 1:
            # if moving_average_200 < moving_average_50 then intersection has occurred
            if moving_average_200[i] < moving_average_50[i]:
                # this is golden cross too
                if moving_average_50[len(moving_average_200)-1] > moving_average_50[i]:
                    golden_cross = True
                intersection_points.append(x_axis[i])
                start = 0
        # error has occurred
        else:
            raise Exception
    # plot intersection points on chart
    # for i in range(len(intersection_points)):
    #     plt.plot([intersection_points[i][0]], [intersection_points[i][1]], color='black', linestyle='dashed', marker='o')

    # save chart for visualizing intersection points
    #fig.savefig("/Users/keeratgrewal/Desktop/StockAnalysis/charts/intersection.jpg")

    mpf.plot(stock.stock_data, type='candle', figratio=(18,10), mav=(50, 200), volume=True,
             vlines=dict(vlines=intersection_points,linewidths=0.5),
             savefig="/Users/keeratgrewal/Desktop/StockAnalysis/charts/intersection.jpg")

    return golden_cross, intersection_points



if __name__ == '__main__':
    main()