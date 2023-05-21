import plotly.graph_objects as go         # To plot the candlestick
import pandas as pd                       # structures and data analysis
import datetime as dt                     # 
import yfinance as yf                     # Yahoo! Finance market data downloader
import seaborn as sns
import scipy.stats as st
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

                                          # Introduce algorithms
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNet
from sklearn.svm import SVR               # Compared with SVC, it is the regression form of SVM.
                                          # Integrate algorithms
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv(r'C:\Users\atom\Desktop\Project 8th Sem\CSV\file.csv')
x = df[['High', 'Low', 'Open', 'Volume']].values  # x features
y = df['Close'].values  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28) # Segment the data
ss = StandardScaler()                                 # Standardize the data set
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

names = ['LinerRegression',
       'Ridge',
       'Lasso',
       'Random Forrest',
       'Support Vector Regression',
       'ElasticNet',
       'XgBoost']

#Define the model.
# cv is the cross-validation idea here.
models = [LinearRegression(),
         RidgeCV(alphas=(0.001,0.1,1),cv=3),
         LassoCV(alphas=(0.001,0.1,1),cv=5),
         RandomForestRegressor(n_estimators=10),
         SVR(),
         ElasticNet(alpha=0.001,max_iter=10000),
         XGBRegressor()]
# Output the R2 scores of all regression models.

#Define the R2 scoring function.
def R2(model,x_train, x_test, y_train, y_test):
        model_fitted = model.fit(x_train,y_train)
        y_pred = model_fitted.predict(x_test)
        score = r2_score(y_test, y_pred)
        return score

#Traverse all models to score.
for name,model in zip(names,models):
        score = R2(model,x_train, x_test, y_train, y_test)
        print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))

#Build a model.

parameters = {
   'kernel': ['linear', 'rbf'],
   'C': [0.1, 0.5,0.9,1,5],
   'gamma': [0.001,0.01,0.1,1]
}

#Use grid search and perform cross validation.
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

print("Optimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)

ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)

st.subheader(f"Predicting the data with Liner Regression")

#Set the canvas.
fig = plt.figure(figsize=(12,6))
#Draw with a red solid line.
plt.plot ( y_test, 'm-o', lw=2, label=u'True values')
#Draw with a green solid line.
plt.plot ( y_predict, 'b--+',  label=u'Predicted value with the SVR algorithm, $R^2$=%.3f' % (model.best_score_))
#Display in a diagram.
plt.legend(loc ='upper left')
# plt.grid(True)
plt.title(u"Stock price prediction with SVR")
plt.ylabel('Price ($)')
plt.xlabel('Days')
plt.show()
st.pyplot(fig)
print(y_test[0])
st.subheader("Testing vs Predicted data of last 50 entries")
df_new = pd.DataFrame(y_test, columns=['Test Data'])
df_new.insert(1,"Predicted Data",y_predict)
df_new['Predicted - Original'] = df_new['Predicted Data'] - df_new['Test Data']
st.write(df_new.tail(50))

