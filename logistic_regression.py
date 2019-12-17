import numpy as np


class LogReg:
    def __init__(self, data_in, data_out):                                              # For preprocessing data.
        self.x = np.array(data_in, dtype=np.float64)                                    # data_in=training input, data_out=training output
        self.y = np.array(data_out, dtype=np.float64)
        self.m = int(data_in.shape[0])                                                  # No. of training exapmles

        try:
            self.col1 = int(data_in.shape[1]) + 1                                       # No. of features plus bias
        except:
            self.col1 = 2                                                               # For one feature training set

        try:
            one = np.ones((1, self.m), dtype=np.float64)
            self.x = np.ndarray.flatten(self.x, order='C')
            self.x = np.concatenate((one, self.x.T), axis=0)
            self.x = np.reshape(self.x, (self.col1, self.m))
        except:
            one = np.ones(self.m, dtype=np.float64)
            self.x = np.concatenate((one, self.x), axis=0)
            self.x = np.reshape(self.x, (self.col1, self.m))

        self.wt = np.random.randn(self.col1)                                            # defining weights matrix OR array
        self.temp = self.wt

    def y_pred(self):
        t = np.dot(self.wt, self.x)
        t= -1*t
        return 1 / (1 + np.e ** t)

    def cost(self):
        return sum((-np.dot(self.y, np.log(self.y_pred()))) - np.dot((1 - self.y), np.log(1 - self.y_pred())))

    def update(self):
        temp = np.dot((self.y_pred() - self.y), self.x.T) / self.m
        self.wt = self.wt - 0.01 *temp

    def train_model(self, epoch):
        for i in range(epoch):
            self.y_pred()
            self.cost()
            self.update()
        return self.wt


