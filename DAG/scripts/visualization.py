import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/cases-by-day.csv")
    df['date_of_interest'] = pd.to_datetime(df['date_of_interest'])
    case_ts = pd.Series(df.CASE_COUNT.values, index=df.date_of_interest)

    file = open("case_count_forecast_20.txt", 'r')
    content = file.read()
    content_list = content.split("\n")
    content_list = content_list[:-1]
    predict_cases = [float(i) for i in content_list]
    predict_cases = list(case_ts.values)[-1:]+predict_cases
    start_date = max(case_ts.index)
    future_dates = [start_date+datetime.timedelta(days=x) for x in range(len(predict_cases))]

    file = open("case_count_rnn.txt", 'r')
    content = file.read()
    content_list = content.split("\n")
    content_list = content_list[:-1]
    rnn_cases = [float(i) for i in content_list]
    rnn_cases = list(case_ts.values)[-1:]+rnn_cases
    # start_date = max(case_ts.index)
    # future_dates = [start_date+datetime.timedelta(days=x) for x in range(len(predict_cases))]

    fig = plt.figure(figsize=(12,6))
    plt.plot(case_ts[-20:].index,case_ts[-20:].values,label='real time series')
    plt.plot(future_dates[:8],predict_cases[:8],linestyle='dashed', label='var forecast')
    plt.plot(future_dates[:8], rnn_cases[:8],linestyle='dashed', label='var forecast')
    plt.legend()
    plt.title("Prediction for Next 7 days")
    plt.savefig("./predictions.png")