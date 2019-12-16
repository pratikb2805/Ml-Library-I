import numpy as np
import pickle as pkl


class Lin_reg:
    def __init__(self, data_in, data_out):  # For preprocessing data.
        self.x = np.array(data_in, dtype=np.float64)  # data_in=training input, data_out=training output
        self.y = np.array(data_out, dtype=np.float64)
        self.m = int(data_in.shape[0])  # No. of training exapmles

        try:
            self.col1 = int(data_in.shape[1]) + 1  # No. of features plus bias
        except:
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

        self.wt = np.random.randn(self.col1)  # defining weights matrix OR array
        self.temp = self.wt

    def y_pred(self):
        return np.matmul(self.wt, self.x)

    def y_cost(self):
        rss = np.zeros(self.col1, dtype=np.float64)
        rss = self.y - self.y_pred()
        rss = np.dot(rss, rss)
        return np.mean(rss)

    def update(self):
        self.temp = np.zeros(self.col1, dtype=np.float64)
        self.temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m
        self.wt = self.wt - 0.01 * self.temp

    def train(self, epoch):
        for i in range(int(epoch)):
            self.y_pred()
            self.y_cost()
            self.update()
        with open('weights.txt', 'wb') as weight:
            write(weight,self.wt)
        print(self.wt)
