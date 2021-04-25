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

l2 = 1000.0
w = np.linalg.solve(l2*np.ones(7) + X.T.dot(X), X.T.dot(Y))
Yhat = X.dot(w)

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - (d1.dot(d1)/d2.dot(d2))
print(f'r-squared: {r2}')  # Nesse caso a L2 Regularization piorou o erro.

plt.scatter(X[:,0], Y)
plt.plot(sorted(X[:,0]), sorted(Yhat), 'red')
plt.show()
