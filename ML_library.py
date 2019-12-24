import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        return np.mean(rss)

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

    def batch_train(self, epoch=100):  # model training function with batch ...
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

    def model_cost_vs_epoch(self, lr, ur, step):  # Plot showing accuracy of model vs epoch attained for taraining
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
        cost = np.mean(np.sqrt(cost))
        return cost

    def cost_cv(self):  # cross-validation cost
        y = np.dot(self.wt, self.xcv)
        cost = np.mean(np.square(y - self.ycv))
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
        self.j = int(np.size(self.y) / self.y.shape[0])
        self.l_rate = float(l_rate)
        try:
            self.col1 = int(data_in.shape[1])  # No. of features plus bias
        except IndexError:
            self.col1 = 1  # For one feature training set
        self.bias = np.zeros((self.j, 1), dtype=np.float64)
        self.wt1 = np.zeros((self.j, self.col1), dtype=np.float64)  # defining weights matrix OR array
        self.temp = np.zeros((self.j, self.col1), dtype=np.float64)
        self.xs = np.array_split(self.x, 2)
        self.ys = np.array_split(self.y, 2)
        self.x = np.transpose(np.array(self.xs[0]))
        self.xcv = np.transpose(np.array(self.xs[1]))
        self.y = np.transpose(np.array(self.ys[0]))
        self.ycv = np.transpose(np.array(self.ys[1]))




    def y_pred_lin(self):  # simple linear hypothesis
        t = np.matmul(self.wt1, self.x)
        for i in range(t.shape[0]):
            t[i, :] += self.bias[i, 0]
        t = t * (-1)
        return 1 / (1 + np.exp(t))

    def cost_lin(self):
        t1 = (-np.dot(self.y, np.log(self.y_pred_lin()).T))
        t2 = - np.dot((1 - self.y), np.log(1 - self.y_pred_lin()).T)
        t3 = (np.square(self.wt1)) * self.lamb / 2 / self.m
        return sum(t1 + t2) + np.sum(t3) + np.sum(np.square(self.bias)) * self.lamb / self.m

    def update_lin(self):
        temp = np.dot((self.y_pred_lin() - self.y), self.x.T) / self.m + self.wt1 * self.lamb / self.m
        temp2 = np.mean(self.y_pred_lin() - self.y, axis=1)
        self.wt1 = self.wt1 - self.l_rate * temp
        self.bias -= temp2 * self.l_rate

    def update_quad(self):
        t1 = np.dot((self.y_pred_quad() - self.y), self.x.T) / self.x.shape[1]
        t2 = np.dot((self.y_pred_quad() - self.y), np.square(self.x.T)) / self.x.shape[1]
        t3 = np.dot((self.y_pred_quad() - self.y), np.square(self.x_prod.T)) / self.x.shape[1]
        temp1 = t1 + self.wt1 * self.lamb / self.x.shape[1]
        temp2 = t2 + self.wt2 * self.lamb / self.x.shape[1]
        temp3 = t3 + self.wt3 * self.lamb / self.x.shape[1]
        self.wt1 = self.wt1 - self.l_rate * temp1
        self.wt2 = self.wt2 - self.l_rate * temp2
        self.wt3 = self.wt3 - self.l_rate * temp3

    def train_model_lin(self, epoch=1000):
        for i in range(epoch):
            self.y_pred_lin()
            self.cost_lin()
            self.update_lin()
        return self.wt1

    def batch_train_lin(self, epoch=100):
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
                self.train_model_lin(10)

    def batch_train_quad(self, epoch):
        n = int(self.m / 50) + 1
        self.x_split1 = np.array_split(self.x, n, axis=1)
        self.y_split1 = np.array_split(self.y, n, axis=1)
        self.x_prod_split = np.array_split(self.x_prod, n, axis=1)
        for j in range(int(epoch)):
            for i in range(n):
                self.x = np.array(self.x_split1[i])
                self.y = np.array(self.y_split1[i])
                self.x_prod = np.array(self.x_prod[i])
                self.train_model_quad(10)

    def train_model_quad(self, epoch):
        for i in range(epoch):
            self.y_pred_quad()
            self.cost_quad()
            self.update_quad()
        print(self.wt1)
        print(self.wt2)
        print(self.wt3)

    def test_model(self, test_data_in, test_data_out):
        self.x_test = np.array(test_data_in,
                               dtype=np.float64)  # test_data_in=training input, test_data_out=training output
        self.y_test = np.array(test_data_out, dtype=np.float64).T
        self.m2 = int(test_data_in.shape[0])  # No. of training exapmles
        self.j2 = int(np.size(self.y_test) / self.y_test.shape[1])
        try:
            self.col2 = int(test_data_in.shape[1]) + 1  # No. of features plus bias
        except IndexError:
            self.col2 = 2  # For one feature training set

        try:
            one = np.ones((1, self.m2), dtype=np.float64)
            self.x_test = np.ndarray.flatten(self.x_test, order='C')
            self.x_test = np.concatenate((one, self.x_test.T), axis=0)
            self.x_test = np.reshape(self.x_test, (self.col2, self.m2))
        except:
            one = np.ones(self.m2, dtype=np.float64)
            self.x_test = np.concatenate((one, self.x_test), axis=0)
            self.x_test = np.reshape(self.x_test, (self.col2, self.m2))
        self.x_prod_test = np.prod(self.x_test, axis=0)
        self.x_prod_test = np.reshape(self.x_prod_test, (1, self.m2))

    def y_test_pred(self):
        try:
            x1 = np.matmul(self.wt1, self.x_test)
            x2 = np.matmul(self.wt2, np.square(self.x_test))
            x3 = np.matmul(self.wt3, self.x_prod_test)
            t = x1 + x2 + x3
        except:
            t = np.matmul(self.wt1, self.x_test)
        return 1 / (1 + np.e ** -t)

    def test_accu(self):
        accu = -1 * np.mean(np.abs(self.y_test_pred() - self.y_test)) * 100 + 100
        print(accu)
        return accu

    def model_accu_vs_epoch(self, lr, ur, step):  # plots learning curve depicting accuray of model
        epoch = np.arange(int(lr), int(ur), int(step))
        accu = np.zeros(epoch.shape[0], dtype=float)
        try:
            for i in range(epoch.shape[0]):
                self.batch_train_quad(epoch[i])
                accu[i] = np.mean(np.abs(self.y_pred_quad() - self.y)) * (-100) + 100
        except:
            for i in range(epoch.shape[0]):
                self.batch_train_lin(epoch[i])
                accu[i] = np.mean(np.abs(self.y_pred_lin() - self.y)) * (-100) + 100

        plt.gcf().canvas.set_window_title('Learning_Curve')
        plt.xlabel = 'Epoch'
        plt.ylabel = 'Accuracy'
        plt.plot(epoch, accu)

    def cost_cv_lin(self):  # cross validation cost on quadratic hypothesis
        y = np.dot(self.wt1, self.xcv)
        cost = np.mean(np.square(y - self.ycv))
        return cost

    def cost_cv_quad(self):  # cross validation cost on quadratic hypothesis
        x1 = np.dot(self.wt1, self.xcv)
        x2 = np.dot(self.wt2, np.square(self.xcv))
        x3 = np.dot(self.wt3, self.x_prodcv)
        cost = np.mean(np.square(x1 + x2 + x3 - self.ycv))
        return cost
