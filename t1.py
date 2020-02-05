# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:39:59 2019

@author: kabir
"""

import pandas as pd
import seaborn as sns
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\MCI_2014_to_2018.csv")

test = df.loc[df["MCI"]=="Theft Over", df.columns[[0, 1, 4]] ] #4 = occurencedate
test["occurrencedate"] = pd.to_datetime(test["occurrencedate"])#.astype(int)/10**17
test = test.sort_values('occurrencedate').drop('occurrencedate', axis='columns')



dataset = test.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

X = dataset[:-1]
Y = dataset[1:]

nrow = len(X)

# reshape into X=t and Y=t+1
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(nrow, 1, 2)))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=10, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()