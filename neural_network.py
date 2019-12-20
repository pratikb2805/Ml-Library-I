#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd


class layers():
    def __init__(self, neu, r, c):
        self.neu = int(neu)
        self.wt = np.random.randn(r, c)
        self.a = np.empty((self.neu, 1), dtype=np.float64)
        self.z = np.empty((self.neu, 1), dtype=np.float64)
        self.error = np.empty((self.neu, 1), dtype=np.float64)
        self.temp = np.empty((r, c), dtype=np.float64)
        self.bias = np.random.randn(self.neu, 1)
        self.temp_bias = np.zeros((self.neu, 1), dtype=np.float64)


class network:
    def __init__(self, data_in, data_out, layers_list, l_rate=0.001):
        self.x = np.array(data_in) / 100
        self.y = np.array(data_out)
        xx = np.array_split(self.x.T, 2)
        self.xt = xx[0].T
        self.xc = xx[1].T
        yy = np.array_split(self.y.T, 2)
        self.yt = yy[0].T
        self.yc = yy[1].T
        self.l = len(li)
        self.ly = layers_list
        self.l_rate = float(l_rate)
        try:
            self.n = self.xt.shape[1]
        except:
            self.n = self.xt.shape[0]
        self.m = int(self.n / 100)
        self.x_split = np.array_split(self.xt.T, self.m)
        self.y_split = np.array_split(self.yt.T, self.m)

    def sigm(self, x):
        return 1 / (1 + np.e ** -x)

    def forward(self):
        for i in range(self.l):
            if i == 0:
                self.ly[i].z = np.matmul(self.ly[i].wt, self.xt) + self.ly[i].bias
                self.ly[i].a = self.sigm(self.ly[i].z)
            else:
                self.ly[i].z = np.matmul(self.ly[i].wt, self.ly[i - 1].a ) + self.ly[i].bias
                self.ly[i].a = self.sigm(self.ly[i].z)

    def cost(self):
        return np.mean((np.dot(self.yt, np.log(self.ly[-1].a).T)) + np.dot(1 - self.yt, np.log(1 - self.ly[-1].a).T))

    def backprop(self):
        for i in range(self.l - 1, -1, -1):
            if i != self.l - 1:
                self.ly[i].error = np.dot(self.ly[i + 1].wt.T, self.ly[i + 1].a)
            else:
                self.ly[i].error = self.ly[i].a - self.yt

    def update(self):
        for i in range(self.l - 1, -1, -1):
            if i != 0:
                self.ly[i].temp = np.matmul(self.ly[i].error, self.ly[i - 1].a.T)
            else:
                self.ly[i].temp = np.matmul(self.ly[i].error, self.xt.T)

            self.ly[i].temp_bias = np.mean(self.ly[i].error, axis=1)

        for i in range(self.l - 1, -1, -1):
            self.ly[i].wt = self.ly[i].wt - self.ly[i].temp * self.l_rate
            self.ly[i].bias =  self.ly[i].bias-self.ly[i].temp_bias * self.l_rate

    def train(self, epoch):
        try:
            epoch = int(int(epoch) / self.m)
            r = int(self.m)
        except:
            epoch = int(epoch)
            r = int(1)
        for i in range(int(epoch)):
            for j in range(int(r)):
                self.xt = self.x_split[i].T
                self.yt = self.y_split[i].T
                self.forward()
                self.cost()
                self.backprop()
                self.update()

    def implement(self, x_data):  # just one sample at a time
        x_data = np.array(x_data) / 100
        l = int(x_data.shape[0])
        x_data = np.expand_dims(x_data, 0)
        x_data = x_data.T
        self.xi = x_data
        for i in range(self.l):
            if i == 0:
                self.ly[i].z = np.dot(self.ly[i].wt, x_data) + self.ly[i].bias
                self.ly[i].a = self.sigm(self.ly[i].z)
            else:
                self.ly[i].z = np.dot(self.ly[i].wt, self.ly[i - 1].a) + self.ly[i].bias
                self.ly[i].a = self.sigm(self.ly[i].z)
        print(self.ly[-1].a)


data = pd.read_csv("mnist_train_small.csv", header=None)
data.dropna()
d = np.array(data)
y = d[:, [0]]
x = np.delete(d, 0, 1)
z = np.eye(10)
yy = np.empty((y.shape[0], 10), dtype=np.float64)
for i in range(y.shape[0]):
    for j in range(10):
        if y[i] == j + 1:
            yy[i, :] = z[j, :]
        else:
            pass

li = []
li.append(layers(4, 4, 784))
li.append(layers(4, 4, 4))
li.append(layers(10, 10, 4))


model = network(x.T, y.T, li)
model.train(50)
model.implement(np.random.randn(784))

