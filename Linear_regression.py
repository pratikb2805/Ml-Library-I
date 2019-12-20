import matplotlib.pyplot as plt
import numpy as np


class LinReg:
    def __init__(self, data_in, data_out, lamb=0):  # For pre-processing data.
        self.x = np.array(data_in, dtype=np.float64)  # data_in=training input,
        self.y = np.array(data_out, dtype=np.float64)  # data_out=training output
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
        self.temp = np.empty(self.col1, dtype=np.float64)
        self.xs = np.array_split(self.x, 2, axis=1)
        self.ys = np.array_split(self.y, 2)
        self.x = self.xs[0]
        self.xcv = self.xs[0]
        self.y = self.ys[0]
        self.ycv = self.ys[0]

    def y_pred(self):  # prediction of output
        return np.matmul(self.wt, self.x)

    def y_cost(self):  # cost function
        rss = np.zeros(self.col1, dtype=np.float64)
        rss = self.y - self.y_pred()
        rss = np.dot(rss, rss.T) + self.lamb * np.sum(np.square(self.wt)) / 2
        return np.mean(rss)

    def update(self):  # gradient descent
        self.temp = np.zeros(self.col1, dtype=np.float64)
        self.temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m + self.lamb * np.sum(self.wt) / self.m
        self.wt = self.wt - 0.01 * self.temp

    def train(self, epoch=1000):  # model training function with traditional..
        for j in range(int(epoch)):  # gradient descent
            self.y_pred()
            self.y_cost()
            self.update()
        return self.wt

    def batch_train(self, epoch):  # model training function with batch ...
        n = int(self.m / 50) + 1 # gradient descent
        self.x_split = np.array_split(self.x, n, axis=1)
        try:
            self.y_split = np.array_split(self.y, n, axis=1)
        except:
            self.y_split = np.array_split(self.y, n)
        for j in range(epoch):
            for i in range(n):
                self.x = np.array(self.x_split[i])
                self.y = np.array(self.y_split[i])
                self.train(10)

    def model_accu_vs_epoch(self):  # Plot showing accuracy of model vs epoch attained for taraining
        epoch = np.arange(100, 10000, 500)
        accu = np.zeros(epoch.shape[0], dtype=float)
        for i in range(epoch.shape[0]):
            self.train(epoch[i])
            accu[i] = 100 - np.mean(np.abs(self.y_pred() - self.y) // np.abs(self.y)) * 100
        plt.xlabel = 'Epoch'
        plt.ylabel = 'Accuracy'
        plt.plot(epoch, accu)

    def test(self, test_data_in, test_data_out):
        self.x_test = np.array(test_data_in)
        self.y_test = np.array(test_data_out)
        try:
            one = np.ones((1, self.m), dtype=np.float64)
            self.x_test = np.ndarray.flatten(self.x_test, order='C')
            self.x_test = np.concatenate((one, self.x_test.T), axis=0)
            self.x_test = np.reshape(self.x_test, (self.col1, test_data_in.shape[0]))
        except:
            one = np.ones(test_data_in.shape[0], dtype=np.float64)
            self.x_test = np.concatenate((one, self.x_test), axis=0)
            self.x_test = np.reshape(self.x_test, (self.col1, test_data_in.shape[0]))

    def y_pred_test(self):
        return np.matmul(self.wt, self.x_test)

    def model_accu_test(self):
        cost = np.sum(np.square(self.y_test - self.y_pred_test()) // self.y_test)
        accu = np.mean(np.sqrt(cost) // self.m) * 100 * (-1) + 100
        print("The above model has ", accu, "% accuracy")
        return accu

    def cost_cv(self):                          #cross-validation cost
        y = np.matmul(self.wt, self.xcv)
        cost = np.mean(np.square(y - self.ycv))
        return cost

    def predict1(self,x):
        x=np.array(x)
        one=np.ones(1,np.float64)
        x=np.concatenate((one,x),axis=1)
        return np.matmul(self.wt,x.T)


x=np.arange(10)
y=x+1
m=LinReg(x,y)
m.batch_train(20)
print(m.cost_cv())
m.model_accu_vs_epoch()
