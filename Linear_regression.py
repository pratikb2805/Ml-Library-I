import pickle as pkl
import numpy as np
class Lin_reg:
    def __init__(self, data_in, data_out):                #data_in=training input,
        self.x = np.array(data_in)
        self.y = np.array(data_out)
        self.row1 = np.size(self.x, axis=0)
        self.col1 = np.size(self.x, axis=1)
        self.wt = np.random.randn(1, self.col1)
        one= np.ones(self.row1, 1)
        self.x = np.concatenate((one, self.x), axis=1)
        self.x = np.transpose(self.x)
        self.temp = self.wt

    def y_pred(self):
        return np.matmul(self.wt, self.x)

    def y_cost(self):
        return (0.5 / self.row1) * (np.sum((self.y - self.y_pred ** 2)))

    def update(self):
        for i in range(self.col1):
            self.temp[i] = (1/np.size(self.x[0]))*(np.sum(np.dot(self.y - self.y_pred, self.x[i])))
        self.wt = self.wt - 0.01 * self.temp

    def train(self):
        while self.temp!=0:
            self.y_pred
            self.y_cost()
            self.update()
        with open('wt.pkl', 'wb') as weight:
            pkl.dump(self.wt, weight)
        print(self.cost())

x1=np.array([1,2,3,4,5])
y1=np.array([2,3,4,5,6])
model=Lin_reg(x1,y1)
model.train()











