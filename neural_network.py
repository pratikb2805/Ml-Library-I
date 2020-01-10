import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import warnings

np.seterr(over='ignore')


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


class layers:
    def __init__(self, neu, r, c):
        self.neu = int(neu)
        self.wt = np.random.randn(r, c)
        self.bias = np.random.randn(self.neu)


class network:
    def __init__(self, data_in, data_out, layers_list, l_rate=0.001):
        self.x = np.array(data_in)
        self.y = np.array(data_out)
        xx = np.array_split(self.x.T, 2)
        self.xt = xx[0].T
        self.xc = xx[1].T
        yy = np.array_split(self.y.T, 2)
        self.yt = yy[0].T
        self.yc = yy[1].T
        self.l = int(len(li))
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
        x = np.clip(x, -30, 30)
        return 1 / (1 + np.e ** -x)

    def der_sigm(self, x):
        return sigm(x) / (1 - sigm(x))

    def forward(self):
        for i in range(self.l):
            self.ly[i].z = np.empty((self.ly[i].neu, self.xt.shape[1]), dtype=np.float64)
            self.ly[i].a = np.empty((self.ly[i].neu, self.xt.shape[1]), dtype=np.float64)

        for i in range(self.l):
            if i == 0:
                for j in range(int(self.ly[i].neu)):
                    self.ly[i].z[j, :] = np.dot(self.ly[i].wt, self.xt)[j, :] + self.ly[i].bias[j]
            else:
                for j in range(int(self.ly[i].neu)):
                    self.ly[i].z[j, :] = np.dot(self.ly[i].wt, self.ly[i - 1].a)[j, :] + self.ly[i].bias[j]
            self.ly[i].a = self.sigm(self.ly[i].z)

    def cost(self):
        return np.sum((np.dot(self.yt, np.log(self.ly[-1].a).T)) + np.dot(1 - self.yt, np.log(1 - self.ly[-1].a).T))

    def backprop(self):
        for i in range(self.l - 1, -1, -1):
            if i != self.l - 1:
                self.ly[i].error = np.dot(self.ly[i + 1].wt.T, self.ly[i + 1].a)
            else:
                self.ly[i].error = self.ly[i].a - self.yt

    def update(self):
        for i in range(self.l - 1, -1, -1):
            if i != 0:
                temp = np.dot(self.ly[i].error, self.ly[i - 1].a.T)
            else:
                temp = np.dot(self.ly[i].error, self.xt.T)
            temp_bias = np.mean(self.ly[i].error, axis=1)
            self.ly[i].wt = self.ly[i].wt - temp * self.l_rate
            self.ly[i].bias = self.ly[i].bias - np.ndarray.flatten(temp_bias) * self.l_rate

    def train(self, epoch):
        for j in range(int(epoch)):
            self.forward()
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

    def train2(self):
        self.forward()
        self.backprop()
        self.update()
        while np.mean(np.abs(self.ly[-1].error)) > (0.3):
            self.forward()
            self.backprop()
            self.update()
