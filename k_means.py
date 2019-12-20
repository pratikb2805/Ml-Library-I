import csv

import numpy as np


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
        k1.reshape(1, 1)
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
                d = 0
                d = np.mean(np.square(self.data[i, :] - self.centre[j, :]))
                di.append(d)
            di = np.array(di)
            ind = int(self.min_ind(di))
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

    def load(self):
        filename = []
        for i in range(self.k):
            filename.append("cluster" + str(i) + ".csv")
        for i in range(self.k):
            with open(filename[i], 'wb', newline=' ') as f:
                writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerow(self.li[i])
