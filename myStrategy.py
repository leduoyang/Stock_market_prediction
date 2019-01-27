import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pdb
from sklearn.tree import export_graphviz
import os
def normalize(data):
    h , w = data.shape
    for col in range(w):
    	data[:,col] = (data[:,col] - np.min(data[:,col])) / (np.max(data[:,col]) - np.min(data[:,col]))
    return data

def feature_of_today(dOhlcv,columns,period):
	num , dim= len(dOhlcv) , len(columns)
	data = np.zeros((num,dim))
	for d in range(dim):
		data[:,d] = dOhlcv[columns[d]]
	data = normalize(data)
	fot = np.zeros((1,period*dim))
	fot[0,:] = data[num-period:num,:].flatten()
	return fot

def dataProcessing(dOhlcv,columns,period,fee):
    num , dim= len(dOhlcv) , len(columns)
    data = np.zeros((num,dim))
    for d in range(dim):
    	data[:,d] = dOhlcv[columns[d]]

    step = 20
    X = np.zeros((num-period-step+1,period*dim))
    y = np.zeros((num-period-step+1,))
    for n in range(num-period-step+1):
    	X[n,:] = data[n:n+period,:].flatten()
    	momentum = data[(n+period-1):(n+period-1)+step,0]
    	T4b , T4s = np.percentile(momentum, 25) , np.percentile(momentum, 60)
    	if data[n+period-1,0] > T4s+ fee :
    		y[n] = -1
    	elif data[n+period-1,0] + fee < T4b :
    		y[n]= 1

    return X,y

def getTechIndicator(df,period):
    df.loc[:,'ma'] = np.array(talib.MA(df['open'],timeperiod=period*2))
    df.loc[:,'ma'] = df['ma'].fillna(df['ma'].mean())
    
    df.loc[:,'macd'], df.loc[:,'macdsignal'], df.loc[:,'macdhist'] = talib.MACD(df['open'], fastperiod=period, slowperiod=period*2)
    df.loc[:,'macd']= df['macd'].fillna(df['macd'].mean())
    df.loc[:,'macdsignal']= df['macdsignal'].fillna(df['macdsignal'].mean())
    df.loc[:,'macdhist']= df['macdhist'].fillna(df['macdhist'].mean())
    
    df.loc[:,'rsi'] = talib.RSI(df['open'],timeperiod=period)
    df.loc[:,'rsi'] = df['rsi'].fillna(df['rsi'].mean())
    
    df.loc[:,'obv'] = talib.OBV(df['open'], df['volume'])
    df.loc[:,'obv'] = df['obv'].fillna(df['obv'].mean())  
    return df

def myStrategy(dailyOhlcvFile, minutelyOhlcvFile, openPrice):
	period = 21
	fee = 100
	dOhlcv = dailyOhlcvFile
	mOhlcv = minutelyOhlcvFile
	dOhlcv.append(dOhlcv.tail(1), ignore_index=True)
	dOhlcv[dOhlcv.shape[0]-1,"open"] = openPrice
	dOhlcv = getTechIndicator(dOhlcv,period)
	columns = ['open','ma','macd','macdsignal','macdhist','rsi','obv']
	period = 1
	X, y = dataProcessing(dOhlcv,columns,period,fee)
	X = normalize(X)    
	X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state=42)
    
	clf = RandomForestClassifier(n_estimators=100, max_depth=13,random_state=0)
	clf.fit(X, y)
    
	importances = clf.feature_importances_
	std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
	indices = np.argsort(importances)[::-1]
    
	# Print the feature ranking
	print("Feature ranking:")
    
	for f in range(X.shape[1]):
		print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
    
	# Plot the feature importances of the forest
	plt.figure(figsize=(24, 12))
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), columns)
	plt.xlim([-1, X.shape[1]])
	plt.show()    

	export_graphviz(clf.estimators_[5],
                feature_names=columns,
                filled=True,
                rounded=True)
	os.system('dot -Tpng tree.dot -o tree.png')
	#print('Acc on validation set: ',clf.score(X_val,y_val))
	today_input = feature_of_today(dOhlcv,columns,period)
	action = clf.predict(today_input)
	pdb.set_trace()
	return int(action[0])
