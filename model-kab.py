# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:39:59 2019

@author: kabir
"""

import pandas as pd
import seaborn as sns
import numpy as np
import math
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Activation, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\MCI_2014_to_2018.csv")

test = df.loc[df["MCI"]=="Theft Over", df.columns[[0, 1, 4]] ] #4 = occurencedate
test["occurrencedate"] = pd.to_datetime(test["occurrencedate"])#.astype(int)/10**17
test = test.sort_values('occurrencedate').drop('occurrencedate', axis='columns')

sns.scatterplot(test["X"], test["Y"])

dataset = test.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
sns.scatterplot(dataset[:, 0], dataset[:, 1])


# test model 1
X = dataset[:-1]
Y = dataset[1:]
Y_og = Y
nrow = len(X)

# reshape into X=t and Y=t+1
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(2, input_shape=(1, 2), return_sequences=True))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=5, batch_size=1, verbose=1)
# make predictions
predict = model.predict(X)
# invert predictions
#predict = scaler.inverse_transform(np.reshape(predict, (predict.shape[0], 2)))
predict = np.reshape(predict, (predict.shape[0], 2))
# calculate root mean squared error
score = math.sqrt(mean_squared_error(Y_og, predict))
print('Train Score: %.2f RMSE' % (score))
diff = Y_og - predict
sns.scatterplot(diff[:, 0], diff[:, 1])


#test 2
start = Input(shape=(1,2))
node = LSTM(4)(start)
end = Dense(2)(node)
model = Model(inputs=start, outputs=end)
model.compile(loss='mean_squared_error', optimizer='adam')
Y = np.reshape(Y, (nrow, 2))
model.fit(X, Y, epochs=5, batch_size=1, verbose=1)

Y
predict
model.summary()
