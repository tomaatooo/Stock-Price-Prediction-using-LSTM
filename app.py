import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end=date.today()

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','GOOGL')
df=yf.download(user_input,start,end)


st.subheader('Data From 2010-2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(10,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(10,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(10,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data_train=pd.DataFrame(df.Close[0:int(len(df)*0.8)])
data_test=pd.DataFrame(df.Close[int(len(df)*0.8):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)

model=load_model('keras_model.h5')


past_100_days=data_train.tail(100)

final_df=pd.concat([past_100_days,data_test],ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Original vs Predicted')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
