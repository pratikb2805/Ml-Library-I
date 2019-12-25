import matplotlib.pyplot as plt
import numpy as np

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

    def y_pred(self):  # simple linear hypothesis
        t = np.matmul(self.wt1, self.x)
        for i in range(t.shape[0]):
            t[i, :] += self.bias[i, 0]
        t = t * (-1)
        return 1 / (1 + np.exp(t))

    def cost(self):
        t1 = (-np.dot(self.y, np.log(self.y_pred()).T))
        t2 = - np.dot((1 - self.y), np.log(1 - self.y_pred()).T)
        t3 = (np.square(self.wt1)) * self.lamb / 2 / self.m
        return sum(t1 + t2) + np.sum(t3) + np.sum(np.square(self.bias)) * self.lamb / self.m

    def update(self):
        temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m + self.wt1 * self.lamb / self.m
        temp2 = np.mean(self.y_pred() - self.y, axis=1)
        self.wt1 = self.wt1 - self.l_rate * temp * self.lamb
        self.bias = self.bias - temp2 * self.l_rate

    def train(self, epoch=1000):
        for i in range(epoch):
            self.y_pred()
            self.cost()
            self.update()
        return self.wt1

    def batch_train(self, epoch=100):
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
        self.x_test = np.array(test_data_in,
                               dtype=np.float64).T  # test_data_in=training input, test_data_out=training output
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

    def cv_pred(self):
        t = np.matmul(self.wt1, self.xcv)
        for i in range(t.shape[0]):
            t[i, :] += self.bias[i, 0]
        return 1 / (1 + np.exp(-t))

    def cost_cv(self):  # cross validation cost on quadratic hypothesis
        cost = np.mean(np.square(self.ycv - self.cv_pred()))
        return cost


