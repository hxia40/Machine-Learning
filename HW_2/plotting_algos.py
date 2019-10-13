import numpy as np
import pandas as pd
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")



path = "/Users/huixia/Documents/ML_2019Fall/HW_2 "

def Stock_Prices():
    df = pd.DataFrame()
    fitpath = path + "/size_fit"
    timepath = path + "/size_time"
    fit_list = [x[0] for x in os.walk(fitpath)]


    for each_dir in fit_list[1:]:

            name = each_dir.split('/Users/huixia/Documents/ML_2019Fall/HW_2//size_fit/')[1]
            print(name)


    #         name = "WIKI/" + ticker.upper()
    #         data = quandl.get(name,
    #                           trim_start = "2000-12-12",
    #                           trim_end = "2014-12-30")
    #         data[ticker.upper()] = data["Adj. Close"]
    #         df = pd.concat([df, data[ticker.upper()]], axis = 1)
    #
    #         # ticker_list.append(ticker)