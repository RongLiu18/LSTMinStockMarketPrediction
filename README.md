# LSTMinStockMarketPrediction
LSTM(Long-short term model) is one of the techniques in recurrence neural network(RNN).In this project, LSTM(Long-short term model) will be used to predict the trend of stock martket based on given NASDAQ time series data. 

## LSTM on python
### Step 1: remove irrelevant elements in the dataset such as "$" character
Characters in dataset will disturb the preprocessing and analysis of the data. Therefore it is necessary to remove the disturbing factors in the data.
```
   data=pd.read_csv("HistoricalQuotes.csv")
   del data['Date']
   data['Close/Last']=data['Close/Last'].str.replace('$', '')
   data['Open']=data['Open'].str.replace('$', '')
   data['High']=data['High'].str.replace('$', '')
   data['Low']=data['Low'].str.replace('$', '')
```

### Step 2:set up testing and training dataset for indepenent variables.
Setting up the shapes of tesing and training data in independent variables
```
   def standard_scaler(Xtraining, Xtesting):
    trainSamples= Xtraining.shape
    trainx=Xtraining.shape
    trainy=Xtraining.shape
    testSamples= Xtesting.shape
    testx=Xtesting.shape
    testy=Xtesting.shape
    
    Xtraining = Xtraining.reshape((trainSamples, trainx * trainy))
    Xtesting = Xtesting.reshape((testSamples, testx * testy))
    
    preprocessor = prep.StandardScaler().fit(Xtraining)
    Xtraining = preprocessor.transform(Xtraining)
    Xtesting = preprocessor.transform(Xtesting)
    
    Xtraining = Xtraining.reshape((trainSamples, trainx, trainy))
    Xtesting = Xtesting.reshape((testSamples, testx, testy))
    
    return Xtraining, Xtesting
 ```
 
### Step 3:preprocess data
Setting up training dataset based on the values of independent variables according to the dataset
```
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
```
### Step 4: build LSTM model
```
def build_model():
    model = Sequential()
    
    model.add(LSTM(units=100,input_shape=(10,1),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    
    return model
```
### Step 5: print out the results of analysis and visualize the predicted trend of stock
```
trainScore = model.evaluate(Xtraining, Ytraining, verbose=1)
print('Train Score: ', (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(Xtesting, Ytesting, verbose=1)
print('Test Score: ' ,(testScore[0], math.sqrt(testScore[0])))   

diff = []
ratio = []

pred = model.predict(Xtesting)
for u in range(len(Ytesting)):
    pred2 = pred[u][0]
    ratio.append((Ytesting[u] / pred2) - 1)
    diff.append(abs(Ytesting[u] - pred2))
       
plt.plot(pred, color='red')
plt.plot(Ytesting, color='green')
plt.legend(loc='upper left')
plt.show()
```


