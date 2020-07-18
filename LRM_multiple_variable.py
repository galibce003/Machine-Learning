import pandas as pd
from sklearn import linear_model


#Loading dataset
df = pd.read_csv('C:/Users/Mehedi Hassan Galib/Desktop/Python/gf.csv')
print(df.shape)  #Return the obs. number of the dataset


#Calculating the median of column "Age"
import math
median_age = math.floor(df.Age.median())
print(median_age)



#Add the median to the missing value
df.Age = df.Age.fillna(median_age)
print(df)


#Linear Regression Model
reg = linear_model.LinearRegression()
y = reg.fit(df[['Area','Bedrooms','Age']],df.Price)
print(reg.coef_)
print(reg.intercept_)
p= reg.predict([[6000,6,8]])
print(p)


















