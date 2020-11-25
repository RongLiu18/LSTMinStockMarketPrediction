# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:26:25 2020

@author: lrzma
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import time
import math


data=pd.read_csv("HistoricalQuotes.csv")
del data['Date']
data.head()
"""
data['Close/Last']=data['Close/Last'].str.replace('$', '')
data['Open']=data['Open'].str.replace('$', '')
data['High']=data['High'].str.replace('$', '')
data['Low']=data['Low'].str.replace('$', '')
"""

def standard_scaler(Xtraining, Xtesting):
    trainSamples= Xtraining.shape
    trainx=Xtraining.shape
    trainy=Xtraining.shape
    testSamples= Xtesting.shape
    testx=Xtesting.shape
    testy=Xtesting.shape
    
    Xtraining = Xtraining.reshape(trainSamples, trainx * trainy)
    Xtesting = Xtesting.reshape(testSamples, testx * testy)
    
    preprocessor = prep.StandardScaler().fit(Xtraining)
    Xtraining = preprocessor.transform(Xtraining)
    Xtesting = preprocessor.transform(Xtesting)
    
    Xtraining = Xtraining.reshape(trainSamples, trainx, trainy)
    Xtesting = Xtesting.reshape(testSamples, testx, testy)

    return Xtraining, Xtesting

def preprocess_data(stocks, sequenceLength2):
    AmountofFeatures = len(stocks.columns)
    data = stocks.values
    
    sequenceLength = sequenceLength2 + 1
    result = []
    for index in range(len(data) - sequenceLength):
        result.append(data[index : index + sequenceLength])
        
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]
    train  = standard_scaler(train, result)
    result = standard_scaler(train, result)
    
    Xtraining = train[:, : -1]
    Ytraining = train[:, -1][: ,-1]
    Xtesting = result[int(row) :, : -1]
    Ytesting = result[int(row) :, -1][ : ,-1]

    Xtraining = np.reshape(Xtraining, (Xtraining.shape[0], Xtraining.shape[1], AmountofFeatures))
    Xtesting = np.reshape(Xtesting, (Xtesting.shape[0], Xtesting.shape[1], AmountofFeatures))  

    return Xtraining, Ytraining, Xtesting, Ytesting

def build_model(layers):
    model = Sequential()
    model.add(LSTM(units=100,input_shape=(10,1),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss="mse", optimizer="rmsprop")
    print(model.summary())
    return model
    



window = 20
Xtraining, Ytraining, Xtesting, Ytesting = preprocess_data(data[::-1], window)
model = build_model([Xtraining.shape[2], window, 1, 1])
model.fit(Xtraining,Ytraining,batch_size=0,epochs=0,validation_split=0.1,verbose=0,sample_weight=None)
print("Xtrain", Xtraining.shape)
print("Ytrain", Ytraining.shape)
print("Xtest", Xtesting.shape)
print("Ytest", Ytesting.shape)

trainScore = model.evaluate(Xtraining, Ytraining, verbose=1)
print('Train Score: ', (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(Xtesting, Ytesting, verbose=1)
print('Test Score: ' ,(testScore[0], math.sqrt(testScore[0])))   


plt.plot(Ytesting, color='red', label='Prediction')
plt.plot(Ytraining, color='green',label='Fact')
plt.legend(loc='upper left')
plt.show()
