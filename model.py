import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


#Data Preprocessing -- Turning data into required useful data

df = pd.read_csv("RELIANCE.BO.csv")
df= df[df.Volume!=0]
df = df.reset_index(drop=True)
df = df.drop(np.where(np.isnan(df['Volume'].to_numpy().astype('float32')))[0])
df = df.reset_index(drop=True)

rows = df.shape[0]
columns = df.shape[1]

df["Open-Close"] = df["Open"] - df["Close"]   # Open - Close
df["High-Low"] = df["High"] - df["Low"]       # High - Low

# 7 day Moving Average
df["7DMA"]=0.000000
for i in range(6,rows):
    df["7DMA"][i] = (df["Close"][i]+df["Close"][i-1]+df["Close"][i-2]+df["Close"][i-3]+df["Close"][i-4]+df["Close"][i-5]+df["Close"][i-6])/7

#21 Day Moving Average
df["21DMA"]=0.000000
for i in range(20,rows):
    df["21DMA"][i] = (df["Close"][i]+df["Close"][i-1]+df["Close"][i-2]+df["Close"][i-3]+df["Close"][i-4]+df["Close"][i-5]+df["Close"][i-6]+df["Close"][i-7]+df["Close"][i-8]+df["Close"][i-9]+df["Close"][i-10]+df["Close"][i-11]+df["Close"][i-12]+df["Close"][i-13]+df["Close"][i-14]+df["Close"][i-15]+df["Close"][i-16]+df["Close"][i-17]+df["Close"][i-18]+df["Close"][i-19]+df["Close"][i-20])/21
    
# 7 day Standard Deviation
import math
df["7DSD"]=0.000000
for i in range(6,rows):
    sum = 0.000000
    for j in range(7):
        sum = sum + (df["Close"][i-j] - df["7DMA"][i])**2
    df["7DSD"][i] = math.sqrt(sum/7)

# Next day Closing Price
# df = df[df.Date!="2021-04-30"]
ndf = df.copy()
ndf = ndf.drop(0)
ndf = ndf.reset_index(drop=True)
df["Next"] = ndf.Close
# df["Next"][6348] = 1994.45

df = df.drop(columns=['Open','High','Close','Adj Close','Low'])
ndf = df[df['21DMA']!=0]
ndf = ndf.drop(rows-1)
ndf = ndf.reset_index(drop=True)


# Forming Dependent and Independent variables x and y

X = ndf[['Date','Volume','Open-Close','High-Low','7DMA','21DMA','7DSD']].to_numpy()
y = ndf['Next'].to_numpy()
x = X[:,1:]



# Data Splitting into Training and Testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
x_train = X_train[:,1:]
x_test = X_test[:,1:]

# Regression Model

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x_train, y_train)


# Prediction

#print(regressor.predict([[253013,-6.55,37,1951.065,1966.46,47.08]]))


# Testing and Prediction based on Model

#y_pred = regressor.predict(x_test)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



# Evaluation of Performnace of Model

#def mean_absolute_percentage_error(y_true, y_pred):
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#def mean_biased_error(y_true, y_pred):
#    return (y_true-y_pred).mean()


#print("R2 score: ",r2_score(y_test, y_pred))
#print("RMSE: ",mean_squared_error(y_test,y_pred,squared=False))
#print("MAPE: ",mean_absolute_percentage_error(y_test,y_pred))
#print("MBE: ",mean_biased_error(y_test,y_pred))


#Graph after Prediction

#date =[]
#for i in range(0,X.shape[0],1000):
#    date.append(X[:,0][i])

#plt.plot(X[:,0],y,color='blue',label='Actual Price')
#plt.plot(X[:,0],regressor.predict(x),color='red',label='Prediction Price')
#plt.title('RELIANCE BSE')
#plt.xticks(date,rotation=90)
#plt.xlabel('Date')
#plt.ylabel('Price of a Stock')
#plt.legend()
#plt.show()



#import requests
#import math
#from datetime import date

#params = {
#    'function': 'TIME_SERIES_DAILY',
#    'symbol': 'RELIANCE.BSE',
#   'outputsize' :'full',
#    'apikey' : 'SD7E3STMW2WO613H'
#   'query': 'New York'
#}

#api_result = requests.get('https://www.alphavantage.co/query', params)
#api_response = api_result.json()
#arr = list(api_response['Time Series (Daily)'].values())
#Open = float(arr[0]['1. open'])
#high = float(arr[0]['2. high'])
#low = float(arr[0]['3. low'])
#close = float(arr[0]['4. close'])
#volume = float(arr[0]['5. volume'])

#o_c = Open - close
#h_l = high - low

#sum_21 = 0.0
#for i in range(21):
##  sum_21 = sum_21 + float(arr[i]['4. close'])
#avg_21 = round(sum_21/21,6)

#sum_7 = 0.0
#for i in range(7):
#  sum_7 = sum_7 + float(arr[i]['4. close'])
#avg_7 = round(sum_7/7,6)

#sum_sd = 0
#for i in range(7):
#  sum_sd = sum_sd + (float(arr[i]['4. close']) - avg_7)**2
#sd_7 = round(math.sqrt(sum_sd/7),6)

#input = [[volume,o_c,h_l,avg_7,avg_21, sd_7]]
#print("Input:",input)

#print("Prediction of next day Closing Price after date",list(api_response['Time Series (Daily)'].keys())[0],": Rs.",round(regressor.predict(input)[0],2))

#saving model to disk
import pickle

pickle.dump(regressor,open('model.pkl','wb'))


