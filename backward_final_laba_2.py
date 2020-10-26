import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def f(x):
    return 2/(1 + np.exp(-x)) - 1 # tan hip

def df(x):
    return 0.5*(1 + x)*(1 - x)

W1 = np.array([[-0.2, -0.4, 0.3], [-0.3, 0.2,0.7]])
W2 = np.array([0.5,0.1])

def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)

def train(epoch):
    global W2, W1
    lmd = 0.01          # крок навчання
    N = 10000          # кількість ітерацій
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # ипадковий вибір вхідного сигрналу

        y, out = go_forward(x[0:3])             # прохід по НМ

        e = y - x[-1]                           # помилка

        delta = e*df(y)                         # Градієнт

        W2[0] = W2[0] - lmd * delta * out[0]    # коригування вагів
        W2[1] = W2[1] - lmd * delta * out[1]    # коригування вагів

        delta2 = W2*delta*df(out)               # вектор градієнтів

        # коригування звязків 2 шару
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd

# дані
data1 = pd.read_csv(r"D:\Study\master_final\H\data3.csv", sep=',')

epoch = np.column_stack((data1['x1'],data1['x2'],data1['y']))
#print(epoch)

train(epoch)        # старт навчання

# перевірка отриманих результатів
z=[]
#print(z)
for x in epoch:
    y, out = go_forward(x[0:3])

    print(f"Значення НМ: {y} => {x[-1]}")
    #math.floor(y)
    z.append(y)

z=pd.DataFrame(z,columns=['zet'])
z[z>=0.5]=1
z[z<0.5]=0

data1['z']=pd.factorize(z['zet'])[0]
#print(data1)
X1=data1['x1']
X2=data1['x2']
z=data1['z']
plt.scatter(X1[z == 1],  X2[z == 1], s=4, c='blue', label=u'клас: 1')
plt.scatter(X1[z == 0], X2[z == 0], s=2, c='red', label=u'клас: 0')

plt.show()

