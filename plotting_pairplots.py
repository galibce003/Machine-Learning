import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Loading datasets
df = pd.read_csv('../Python/insurance.csv')
print(df.head(15))


#Plotting the pairplots
sns.pairplot(df, hue = 'region')
plt.show()
