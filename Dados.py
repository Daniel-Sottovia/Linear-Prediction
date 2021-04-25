import numpy as np 
import pandas 
import matplotlib.pyplot as plt 

df = pandas.read_csv("world-happiness-report-2021.csv")
df['ones'] = 1

"""print(df.info())
print(df.columns)"""

x = df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'ones']]
y = df['Ladder score']

# Criando os numpy
X = np.array(x)
Y = np.array(y)

print(X.shape)
print(Y.shape)

w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat = X.dot(w)

plt.scatter(X[:,1] , Y)
plt.plot(sorted(X[:,1]),sorted(Yhat), 'green')
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - (d1.dot(d1)/d2.dot(d2))
print(f'r-squared: {r2}')