import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class Lin_reg:
    def __init__(self, data_in, data_out):  # data_in=training input,
        self.x = np.array(data_in,dtype=np.float64)
        self.y = np.array(data_out,dtype=np.float64)
        self.m=int(data_in.shape[0])

        try:
            self.col1=int(data_in.shape[1])+1
        except:
            self.col1=2

        try:
            one=np.ones((1,self.m),dtype=np.float64)
            self.x=np.ndarray.flatten(self.x,order='C')
            self.x=np.conacatenate((one,self.x.T),axis=0)
            self.x=np.reshape(self.x,(self.col1,self.m))
        except:
            one = np.ones(self.m, dtype=np.float64)
            self.x = np.concatenate((one, self.x), axis=0)
            self.x = np.reshape(self.x, (self.col1, self.m))
        self.wt=np.random.randn(self.col1)
        self.temp=self.wt

    def y_pred(self):
        return np.matmul(self.wt, self.x)

    def y_cost(self):
        rss=np.zeros(self.col1,dtype=np.float64)
        rss=self.y-self.y_pred()
        rss=np.dot(rss,rss)
        return np.mean(rss)

    def update(self):
        self.temp=np.zeros(self.col1,dtype=np.float64)
        self.temp = np.dot((self.y_pred()-self.y), self.x.T)/self.m
        self.wt = self.wt - 0.01 * self.temp

    def train(self, epoch):
        for i in range(int(epoch)):
            self.y_pred()
            self.y_cost()
            self.update()
        with open('wt.pkl', 'wb') as weight:
            pkl.dump(self.wt, weight)
        print(self.y_cost())
        print(self.wt)




