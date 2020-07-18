import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/Mehedi Hassan Galib/Desktop/Python/gf.csv')
print(df.shape)  #Return the obs. number of the dataset


#Finding out the R-squared value
X= df['Area'].values
Y= df['Price'].values
m = len(X)
X = X.reshape((m,1))
reg = LinearRegression()
reg = reg.fit(X,Y)
Y_p = reg.predict(X)
rr = reg.score(X,Y)
print(rr)
















