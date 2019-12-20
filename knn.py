import numpy as np
import pandas as pd
class KNN:
    def __init__(self, dataset, query, k):
        self.x = np.array(dataset)
        self.y = np.array(query)
        self.k = int(k)
        try:
            self.row = self.x.shape[0]
            self.col = self.x.shape[1]
        except:
            self.row = self.x.shape[0]
            self.col = 1

        z = np.arange(self.row)
        y = np.zeros(self.row, dtype=float)
        self.dist = np.concatenate((z, y), axis=0)
        self.dist.reshape(self.row,2)
        self.dist = np.transpose(self.dist)

    def di(self):
        for i in range(self.row):
            self.dist[i, 1] = np.sum(np.square(self.x[i, :] - self.y))

    def data_sort(self):
        self.dist = self.dist[self.dist[:, 1].argsort()]

    def out(self):
        self.di()
        self.data_sort()
        ans= np.empty(())
        for i in range(self.k):
             self.x[self.dist[i, 0], :]