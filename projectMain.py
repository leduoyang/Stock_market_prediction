import sys
import numpy as np
import pandas as pd
from myStrategy import myStrategy

dailyOhlcv = pd.read_csv("./ohlcv_daily.csv")
minutelyOhlcv = pd.read_csv("./ohlcv_daily.csv")
evalDays = 1
action = np.zeros((evalDays,1))
openPricev = dailyOhlcv["open"].tail(evalDays).values
for ic in range(evalDays,0,-1):
    dailyOhlcvFile = dailyOhlcv.head(len(dailyOhlcv)-ic)
    dateStr = dailyOhlcvFile.iloc[-1,0]
    #minutelyOhlcvFile = minutelyOhlcv.head((np.where(minutelyOhlcv.iloc[:,0].str.split(expand=True)[0].values==dateStr))[0].max()+1)
    action[evalDays-ic] = myStrategy(dailyOhlcvFile,minutelyOhlcv,openPricev[evalDays-ic])
