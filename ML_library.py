import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.seterr(over='ignore')



np.seterr(divide='ignore', invalid='ignore')


class LinReg:
    def __init__(self, data_in, data_out, lamb=0, l_rate=.01):  # For pre-processing data.
        self.x = np.array(data_in, dtype=np.float64)  # data_in=training input,
        self.y = np.array(data_out, dtype=np.float64)  # data_out=training output
        self.m = int(data_in.shape[0])  # No. of training exapmles
        self.lamb = int(lamb)
        try:
            self.col1 = int(data_in.shape[1])  # No. of features plus bias
        except IndexError:
            self.col1 = 1  # For one feature training set
        self.x = np.transpose(self.x)
        self.y = np.reshape(self.y, (1, self.m))
        self.wt = np.zeros((1, self.col1), dtype=float)  # defining weights matrix OR array
        self.temp = np.empty((1, self.col1), dtype=np.float64)
        self.x = np.ndarray.flatten(self.x)
        self.x = np.reshape(self.x, (self.col1, self.m))

        try:
            self.xs = np.array_split(self.x, 2, axis=1)
        except:
            self.xs = np.array_split(self.x, 2)

        self.ys = np.array_split(self.y, 2, axis=1)
        self.x = np.array(self.xs[0])
        self.xcv = np.array(self.xs[1])
        self.y = np.array(self.ys[0])
        self.ycv = np.array(self.ys[1])
        self.bias = 0.00
        self.l_rate = float(l_rate)

    def y_pred(self):  # prediction of output
        return np.dot(self.wt, self.x) + self.bias

    def y_cost(self):  # cost function
        m = self.lamb / (2 * self.m)
        rss = np.zeros(self.col1, dtype=np.float64)
        rss = self.y - self.y_pred()
        rss = np.sum(np.square(rss)) + np.sum(np.square(self.wt)) * m + (self.bias ** 2) * m
        return np.mean(rss) / 2

    def update(self):  # gradient descent
        self.temp = np.zeros(self.col1, dtype=np.float64)
        self.temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m + self.lamb * np.sum(self.wt) / self.m
        self.wt = self.wt - self.l_rate * self.temp
        self.bias -= np.sum(self.y_pred() - self.y) * self.l_rate / self.m

    def train(self, epoch=1000):  # model training function with traditional..
        for j in range(int(epoch)):  # gradient descent
            self.y_pred()
            self.y_cost()
            self.update()
            self.wt0 = self.wt
        return self.wt

    def batch_train(self, epoch=100):  # model training function with mini batch ...
        n = int(self.m / 50) + 1  # gradient descent

        try:
            self.x_split = np.array_split(self.x, n, axis=1)
        except:
            self.x_split = np.array_split(self.x, n)
        try:
            self.y_split = np.array_split(self.y, n, axis=1)
        except:
            self.y_split = np.array_split(self.y, n)
        for j in range(epoch):
            for i in range(n):
                self.x = np.array(self.x_split[i])
                self.y = np.array(self.y_split[i])
                self.train(10)

    def model_cost_vs_epoch(self, lr, ur, step):  # Plot showing accuracy of model vs epoch attained for training
        self.wt = np.zeros((1, self.col1), dtype=float)
        self.bias = 0.00
        epoch = np.arange(int(lr), int(ur), int(step))
        cost = np.zeros(epoch.shape[0], dtype=float)
        self.train(int(lr))
        for i in range(epoch.shape[0]):
            self.batch_train(int(step))
            cost[i] = self.cost_cv()
        plt.xlabel = 'Epoch'
        plt.ylabel = 'Cost_CV'
        plt.plot(epoch, cost)

    def test(self, test_data_in, test_data_out):
        self.x_test = np.array(test_data_in)
        m2 = int(self.x_test.shape[0])
        self.y_test = np.array(test_data_out)
        self.y_test = np.reshape(self.y_test, (m2, 1))
        self.x_test = self.x_test.T

    def y_pred_test(self):
        return np.dot(self.wt, self.x_test) + self.bias

    def cost_test(self):
        cost = np.mean(np.square(self.y_test - self.y_pred_test()))
        cost = np.mean(cost) / 2
        return cost

    def cost_cv(self):  # cross-validation cost
        y = np.dot(self.wt, self.xcv)
        cost = np.mean(np.square(y - self.ycv)) / 2
        return cost

    def predict1(self, x):
        x = np.array(x)
        m = x.shape[0]
        x = np.reshape(x, (m, 1))
        return np.dot(self.wt, x)


class LogReg:
    def __init__(self, data_in, data_out, lamb=0, l_rate=0.001):  # For preprocessing data.
        self.x = np.array(data_in, dtype=np.float64)  # data_in=training input, data_out=training output
        self.y = np.array(data_out, dtype=np.float64)
        self.m = int(data_in.shape[0])  # No. of training exapmles
        self.lamb = float(lamb)

        self.l_rate = float(l_rate)
        try:
            self.col1 = int(data_in.shape[1])  # No. of features plus bias
        except IndexError:
            self.col1 = 1  # For one feature training set
        self.j = int(np.size(self.y) / self.y.shape[0])
        self.wt1 = np.zeros((self.j, self.col1), dtype=np.float64)  # defining weights matrix OR array
        self.temp = np.zeros((self.j, self.col1), dtype=np.float64)
        self.xs = np.array_split(self.x, 2)
        self.ys = np.array_split(self.y, 2)
        self.x = np.transpose(np.array(self.xs[0]))
        self.xcv = np.transpose(np.array(self.xs[1]))
        self.y = np.transpose(np.array(self.ys[0]))
        self.ycv = np.transpose(np.array(self.ys[1]))

        self.bias = np.zeros((self.j, 1), dtype=np.float64)

    def y_pred(self):  # simple linear hypothesis' prediction
        t = np.matmul(self.wt1, self.x)
        for i in range(t.shape[0]):
            t[i, :] += self.bias[i, 0]
        t = t * (-1)
        return 1 / (1 + np.exp(t))

    def cost(self):
        t1 = (-np.dot(self.y, np.log(self.y_pred()).T)) / self.m
        t2 = - np.dot((1 - self.y), np.log(1 - self.y_pred()).T) / self.m
        t3 = (np.square(self.wt1)) * self.lamb / 2 / self.m
        return sum(t1 + t2) + np.sum(t3) + np.sum(np.square(self.bias)) * self.lamb / self.m

    def update(self):
        temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m + self.wt1 * self.lamb / self.m
        temp2 = np.mean(self.y_pred() - self.y, axis=1)
        self.wt1 = self.wt1 - self.l_rate * temp * self.lamb
        self.bias = self.bias - temp2 * self.l_rate

    def train(self, epoch=1000):  # training function employing batch gradient descent
        for i in range(epoch):
            self.y_pred()
            self.cost()
            self.update()
        return self.wt1

    def batch_train(self, epoch=100):  # training function employing mini-batch gradient descent
        n = int(self.m / 50) + 1
        try:
            self.x_split1 = np.array_split(self.x, n, axis=1)
        except:
            self.x_split1 = np.array_split(self.x, n)
        try:
            self.y_split1 = np.array_split(self.y, n, axis=1)
        except:
            self.y_split1 = np.array_split(self.y, n)
        for j in range(int(epoch)):
            for i in range(n - 1):
                self.x = np.array(self.x_split1[i])
                self.y = np.array(self.y_split1[i])
                self.train(10)

    def test_model(self, test_data_in, test_data_out):
        self.x_test = np.array(test_data_in, dtype=np.float64).T
        # test_data_in=training input, test_data_out=training output
        self.y_test = np.array(test_data_out, dtype=np.float64).T

    def y_test_pred(self):
        t = np.matmul(self.wt1, self.x_test) + self.bias
        return 1 / (1 + np.e ** -t)

    def test_accu(self):
        accu = -1 * np.mean(np.square(self.y_test_pred() - self.y_test)) * 100 + 100
        print(accu)
        return accu

    def model_accu_vs_epoch(self, lr, ur, step):  # plots learning curve depicting accuray of model
        epoch = np.arange(int(lr), int(ur), int(step))
        accu = np.zeros(epoch.shape[0], dtype=float)
        self.batch_train(int(lr))
        accu[0] = self.cost_cv() * (-100) + 100
        for i in range(1, epoch.shape[0], 1):
            self.batch_train(int(step))
            accu[i] = self.cost_cv() * (-100) + 100
        plt.gcf().canvas.set_window_title('Learning_Curve')
        plt.xlabel = 'Epoch'
        plt.ylabel = 'Accuracy'
        plt.plot(epoch, accu)

    def cv_pred(self):  # predicting output over cross validation dataset
        t = np.matmul(self.wt1, self.xcv)
        for i in range(t.shape[0]):
            t[i, :] += self.bias[i, 0]
        return 1 / (1 + np.exp(-t))

    def cost_cv(self):  # cross validation cost
        cost = np.mean(np.square(self.ycv - self.cv_pred())) / 2
        return cost


class KNN:
    def __init__(self, dataset, query):
        self.x = np.array(dataset)
        m = self.x.shape[0]
        self.y = np.array(query)
        try:
            n = self.x.shape[1]
        except:
            n = 1
        if self.x.ndim == 1:  # if dataset is 1d, then making it two dimensional
            self.x = self.x[:, np.newaxis]
        else:
            pass

        try:
            self.row = self.x.shape[0]
            self.col = self.x.shape[1]
        except:
            self.row = self.x.shape[0]
            self.col = 1

        z = np.arange(self.row)[:, np.newaxis]
        y = np.zeros((self.row, 1), dtype=np.float64)
        s = np.concatenate((z, y), axis=1)
        self.dist = s  # creating distance matrix

    def di(self):  # calculates distances
        for i in range(self.row):
            self.dist[i, 1] = np.sum(np.square(self.x[i, :] - self.y))

    def data_sort(self):  # function sorting distances in increasing order
        self.dist = self.dist[self.dist[:, 1].argsort()]

    def out(self, k):  # return k nearest neighbours
        self.di()
        self.k = int(k)
        self.data_sort()
        ans = []
        for i in range(int(k)):
            ind = int(self.dist[i, 0])
            ans.append(ind)
            self.ans = ans
        return ans

    def di2(self):  # mean distance from neighbors
        li = self.ans
        li2 = []
        sum1 = 0
        for i in range(self.k):
            sum1 += self.dist[i, 1]
        sum1 = sum1 / self.k
        return sum1


class K_Means:
    def __init__(self, data, k):
        self.data = np.array(data)
        self.k = int(k)
        self.m = self.data.shape[0]
        try:
            self.col = self.data.shape[1]
        except:
            self.col = 1

        ma = np.max(self.data)
        mi = np.min(self.data)
        self.centre = np.random.randn(self.k, self.col) * (ma - mi)

    def min_ind(self, x):
        k = np.where(x == np.ndarray.min(x))
        k1 = k[0]
        return k1

    def li_init(self):
        li = []
        for i in range(self.k):
            li.append([])
        self.li = li

    def cluster(self):
        for i in range(self.m):
            di = []
            for j in range(self.k):
                d = np.mean(np.square(self.data[i, :] - self.centre[j, :]))
                di.append(d)
            di = np.array(di)
            ind = int(np.argmin(di))
            self.li[ind].append(self.data[i, :])

    def calc_centroid(self):
        for i in range(self.k):
            self.centre[i, :] = np.mean(np.array(self.li[i]), axis=0)

    def train(self, epoch):
        for i in range(int(epoch)):
            self.li_init()
            self.cluster()
            self.calc_centroid()

        return tuple(map(tuple, self.centre))


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
