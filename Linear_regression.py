import matplotlib.pyplot as plt
import numpy as np


class LinReg:
    def __init__(self, data_in, data_out, lamb=0):  # For preprocessing data.
        self.x = np.array(data_in, dtype=np.float64)  # data_in=training input, data_out=training output
        self.y = np.array(data_out, dtype=np.float64)
        self.m = int(data_in.shape[0])  # No. of training exapmles
        self.lamb = int(lamb)
        try:
            self.col1 = int(data_in.shape[1]) + 1  # No. of features plus bias
        except IndexError:
            self.col1 = 2  # For one feature training set

        try:
            one = np.ones((1, self.m), dtype=np.float64)
            self.x = np.ndarray.flatten(self.x, order='C')
            self.x = np.concatenate((one, self.x.T), axis=0)
            self.x = np.reshape(self.x, (self.col1, self.m))
        except:
            one = np.ones(self.m, dtype=np.float64)
            self.x = np.concatenate((one, self.x), axis=0)
            self.x = np.reshape(self.x, (self.col1, self.m))

        self.wt = np.zeros(self.col1, dtype=float)  # defining weights matrix OR array
        self.temp = self.wt
        self.bias = np.arange(10)

    def y_pred(self):
        return np.matmul(self.wt, self.x)

    def y_cost(self):
        rss = np.zeros(self.col1, dtype=np.float64)
        rss = self.y - self.y_pred()
        rss = np.dot(rss, rss) + self.lamb * np.sum(np.square(self.wt))
        return np.mean(rss)

    def update(self):
        self.temp = np.zeros(self.col1, dtype=np.float64)
        self.temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m
        self.wt = self.wt - 0.01 * self.temp

    def train(self, epoch=1000):
        self.wt = self.wt * 0
        for j in range(int(epoch)):
            self.y_pred()
            self.y_cost()
            self.update()
        print(self.wt)

    def train2(self, epoch=1000):
        self.wt = self.wt * 0
        for j in range(int(epoch)):
            self.y_pred()
            self.y_cost()
            self.update()

    def model_accu(self):
        epoch = np.arange(100, 10000, 500)
        accu = np.zeros(epoch.shape[0], dtype=float)
        for i in range(epoch.shape[0]):
            self.train2(epoch[i])
            accu[i] = self.y_cost()
        plt.xlabel = 'Epoch'
        plt.ylabel = 'Cost'
        plt.plot(epoch, accu)

    def test(self, test_data_in, test_data_out):
        if self.wt.all()!=0:
            self.x_test = np.array(test_data_in)
            self.y_test = np.array(test_data_out)
            try:
                one = np.ones((1, self.m), dtype=np.float64)
                self.x_test = np.ndarray.flatten(self.x_test, order='C')
                self.x_test = np.concatenate((one, self.x_test.T), axis=0)
                self.x_test = np.reshape(self.x_test, (self.col1, self.m))
            except:
                one = np.ones(self.m, dtype=np.float64)
                self.x_test = np.concatenate((one, self.x_test), axis=0)
                self.x_test = np.reshape(self.x_test, (self.col1, self.m))
            y_pred_test = np.matmul(self.wt, self.x_test)
            cost = np.mean(np.square(self.y_test - y_pred_test))
            return cost
        else:
            print("Train your model first")

