import pickle as pkl
import numpy as np
class Lin_reg:
    def __init__(self, data_in, data_out):                #data_in=training input,
        self.x = np.array(data_in)
        self.y = np.array(data_out)


        self.row1= int(data_in.shape[0])
        self.col1= int(data_in.shape[0])+1
        self.wt = np.random.randn(1, self.col1)
        self.zero=np.zeros((1, self.col1),dtype=float)
        one= np.ones((self.row1, 1),dtype=float)
        self.x = np.concatenate((one, self.x), axis=1)
        self.x = np.transpose(self.x)
        self.temp = self.wt

    def y_pred(self):
        return np.matmul(self.wt, self.x)

    def y_cost(self):
        return (0.5 / self.row1) * (np.sum(np.square(self.y - self.y_pred ))

    def update(self):
        for i in range(self.col1):
            self.temp[i] = (1/np.size(self.x[0]))*(np.sum(np.dot(self.y - self.y_pred, self.x[i])))
        self.wt = self.wt - 0.01 * self.temp

    def train(self):
        while self.temp.all()!= self.zero.all():
            self.y_pred
            self.y_cost()
            self.update()
        with open('wt.pkl', 'wb') as weight:
            pkl.dump(self.wt, weight)
        print(self.cost())

x1=np.random.randn(5,1)
y1=np.random.randn(1,5)
model=Lin_reg(x1,y1)
model.train()











