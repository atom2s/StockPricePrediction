import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import  data as pdr
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yfin
import openai
import re

yfin.pdr_override()

#""" We can not use chat gpt api as it forbidden so put your own apikey and change the code as you need"""
#open AI stuff
# openai.api_key = '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'
# messages = [ {"role": "system", "content": 
#                   "You are a intelligent assistant."} ]
# #function which returns stock ticker name
# def find(message):
#     if message:
#         messages.append(
#             {"role": "user", "content": message},
#         )
#         chat = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo", messages=messages
#         )
#     reply = chat.choices[0].message.content
#     print(f"ChatGPT: {reply}")
#     messages.append({"role": "assistant", "content": reply})
#     return reply






#ask user to input
start = '2010-01-01'
end = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
#label
st.title('Stock Price Prediction')

ticker= st.text_input('Enter stock ticker of the Company','AAPL')
# user_input = 'find me the  stock ticker name of '+input_only+' from yahoo finance. The name should be with in inverted comma'
# answer = find(user_input)
# ticker = re.findall('"([^"]*)"', answer)[0] #find the data which is with in inverted comma
# st.subheader("The Stock Ticker Name of "+input_only+" is "+ticker)
# st.subheader("Showing the Historical data of "+ticker)

df = pdr.get_data_yahoo(ticker, start, end)
# df.to_csv(r'C:\Users\atom\Desktop\Project 8th Sem\CSV\file.csv')



#Describing Data
st.subheader('Metadata from 1\'st Jan 2010')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price VS Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r', label = 'Closing Price')
plt.ylabel('Price')
plt.xlabel('Year')
plt.legend()
st.pyplot(fig)

#100 Days moving Average
st.subheader('Closing Price VS Time chart with 100 days Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r', label = 'Closing Price')
plt.plot(ma100, 'b', label = '100 days Moving AVG')
plt.ylabel('Price')
plt.xlabel('Year')
plt.legend()
st.pyplot(fig)
#200 Days moving Average
st.subheader('Closing Price VS Time chart with 200 days Moving Average')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'b', label = '100 days Moving AVG')
plt.plot(df.Close, 'r', label = 'Closing Price')
plt.plot(ma200, 'g', label = '200 days Moving AVG')
plt.ylabel('Price')
plt.xlabel('Year')
plt.legend()
st.pyplot(fig)
