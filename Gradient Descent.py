import numpy as np
import pandas
import matplotlib.pyplot as plt

#Tá dando errado, testar esse modo em outro conjunto de dados.

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

w = np.random.randn(7) / np.sqrt(7)

learning_rate = 0.001
costs = []
for t in range(100):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse = delta.dot(delta) / 149
    costs.append(mse)

print(w)
plt.plot(costs)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - (d1.dot(d1)/d2.dot(d2))
print(f'r-squared: {r2}')