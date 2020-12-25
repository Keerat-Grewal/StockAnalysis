from stock import *
import numpy as np
import matplotlib.pyplot as plt
# from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import pandas as pd
import math
import os
import copy
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def main():
    # create a stock
    stock = Stock("EURGBP=X", "max", "1d")

    # get moving average, MACD, and RSI for stock
    stock.get_MA(21)
    # stock.get_MA(200)
    stock.get_MACD()
    stock.get_RSI()

    # get real price for stock
    real_price = stock.stock_data
    # for col in real_price.columns:
    #   print(col)
    # print("\n\n")
    print(real_price["Close"])
    real_price = real_price.drop('Date', axis=1)

    real_price = real_price["Close"].values

    # create scaler for normalization
    scalar = MinMaxScaler(feature_range=(0, 1))

    # setup training and testing set
    X_train, y_train, test_set, training_set = setup_data(stock, scalar)

    # get LSTM model
    # model = create_model(X_train, y_train)
    model = tf.keras.models.load_model('./saved_model/my_model')

    # setting up test set
    new_test_set = []
    for i in range(len(training_set) - 1, len(training_set) - 61, -1):
        new_test_set.append(training_set[i])

    assert len(new_test_set) == 60

    for j in range(len(test_set)):
        new_test_set.append(test_set[j])
    test_set = copy.deepcopy(new_test_set)

    test_set_scaled = scalar.transform(test_set)
    X_test = []
    # y_test = []

    for i in range(60, len(test_set_scaled)):
        component = []
        for j in range(i - 60, i):
            for k in range(len(test_set_scaled[j])):
                component.append(test_set_scaled[j][k])
        # this component will contain all the 10 indicators to predict the current price @ i
        actual_component = []
        counter = 0
        for q in range(60):
            item = component[counter:counter + 10]
            actual_component.append(item)
            counter += 10
        X_test.append(actual_component)
        # y_test.append(test_set_scaled[i, :])

    # assert len(X_test) == len(test_set_scaled) - 60 + 1
    X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # made price predictions
    model_prediction = model.predict(X_test)
    predicted_price = scalar.inverse_transform(model_prediction)[:, 3]

    # save the model
    # export_path_keras = "./{}".format("saved_model/my_model")
    # model.save(export_path_keras)

    print("Real last 30 days")
    print(real_price[-30:])
    print("\n")
    print("Predicted last 30 days")
    print(predicted_price[-30:])
    print("\n")

    # predicted_price = predicted_price[:-1]

    # calculate error
    average_error = 0
    for i in range(len(predicted_price)):
        average_error += abs(predicted_price[-1 * i - 1] - real_price[-1 * i - 1])
    average_error /= len(predicted_price)

    # get accuracy
    # hit = 0
    # total = 0
    # for i in range(len(predicted_price) - 2, -1, -1):
    #     if predicted_price[-1 * i] < predicted_price[1 * i - 2]:
    #         pass
    #     else:
    #         pass

    print(f"Average error: {average_error}")
    # print(f"Predicted range: ({predicted_next_day_open - average_error} , {predicted_next_day_open + average_error})")
    # print(f"Predicted next day open: {predicted_next_day_open}")

    # plot results : Predicted vs. Actual
    plt.plot(predicted_price, color="blue", label="Predicted price")
    plt.plot(real_price[len(real_price) - len(predicted_price):], color="red", label="Actual price")
    plt.savefig("/Users/keeratgrewal/Desktop/StockAnalysis/charts/prediction.jpg")


def create_model(X_train, y_train):
    # create model
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    # model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

    return model


def setup_data(stock, scalar):
    # split up data 90% training, 10% testing
    data = stock.stock_data.dropna(thresh=11)
    data = data.drop('Date', axis=1)
    # split = int(0.9 * len(data))
    # print(split)
    training_set = data.iloc[:5105, :]
    test_set = data.iloc[5105:, :]

    training_set = training_set.values
    test_set = test_set.values

    # print(training_set)
    # print(len(training_set))
    # print(test_set)
    # print(len(test_set))

    # assert len(training_set) + len(test_set) == len(data) - 1

    # normalize training data to make it easier for training process
    training_set_scaled = scalar.fit_transform(training_set)

    X_train = []
    y_train = []

    for i in range(60, len(training_set_scaled)):
        component = []
        for j in range(i - 60, i):
            for k in range(len(training_set_scaled[j])):
                component.append(training_set_scaled[j][k])
        # this component will contain all the 10 indicators to predict the current price @ i
        actual_component = []
        counter = 0
        for q in range(60):
            item = component[counter:counter + 10]
            actual_component.append(item)
            counter += 10
        X_train.append(actual_component)
        y_train.append(training_set_scaled[i, :])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train, test_set, training_set


# -------------------------------------------------------------------------------------------------------------- #
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
    # signal_line = MACD[1]
    MACD_hist = MACD[2]
    # print(MACD_hist)

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
                # print(MACD_hist[i])
                return 2
            else:
                # print(MACD_hist[i])
                return 1
        elif not negative and MACD_hist[i] <= 0:
            if actual_MACD[i] > 0:
                # print(MACD_hist[i])
                return 4
            else:
                # print(MACD_hist[i])
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

    x_axis = stock.stock_data.index[original_length - len(times):]

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
                if moving_average_50[len(moving_average_200) - 1] > moving_average_50[i]:
                    golden_cross = True
                intersection_points.append(x_axis[i])
                start = 1
        # if the moving_average_200 > moving_average_50
        elif start == 1:
            # if moving_average_200 < moving_average_50 then intersection has occurred
            if moving_average_200[i] < moving_average_50[i]:
                # this is golden cross too
                if moving_average_50[len(moving_average_200) - 1] > moving_average_50[i]:
                    golden_cross = True
                intersection_points.append(x_axis[i])
                start = 0
        # error has occurred
        else:
            raise Exception

    # plot intersection points on chart and save chart
    # print(intersection_points)
    current_directory = os.getcwd()
    path = current_directory + "/charts/intersection.jpg"
    mpf.plot(stock.stock_data, type='candle', figratio=(18, 10), mav=(50, 200), volume=True, title=stock.ticker.ticker,
             vlines=dict(vlines=intersection_points, linewidths=0.5),
             savefig=path)

    return golden_cross, intersection_points


if __name__ == '__main__':
    main()