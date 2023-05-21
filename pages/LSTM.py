import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler



st.title('Predicting The Data Using LSTM')
df = pd.read_csv(r"C:\Users\atom\Desktop\Project 8th Sem\CSV\file.csv")


#split data into traing and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Scale down data with min_max_scaler

scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)



#Load my model
model = load_model('Model.h5')
#Testing 
past_100 = data_training.tail(100)
final_df = past_100.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Predict
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'g', label = "Predicted Price")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Next
st.subheader(f"Last predicted Price:{y_predicted[-1]}")
df_new = pd.DataFrame(y_predicted, columns = ["Predicted"])
df_new.insert(1, 'Original', y_test, True)
df_new['Predicted - Original'] = df_new['Predicted']- df_new['Original']
st.subheader("Predicted vs Original of thr last 50 entries")
st.write(df_new.tail(50))
