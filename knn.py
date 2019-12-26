
import numpy as np
import pandas as pd
class KNN:
    def __init__(model, dataset, query):
        model.x = np.array(dataset)
        m = model.x.shape[0]
        model.y = np.array(query)
        try:
            n = model.x.shape[1]
        except:
            n = 1
        if model.x.ndim == 1:
            model.x = model.x[:, np.newaxis]
        else:
            pass

        try:
            model.row = model.x.shape[0]
            model.col = model.x.shape[1]
        except:
            model.row = model.x.shape[0]
            model.col = 1

        z = np.arange(model.row)[:, np.newaxis]
        y = np.zeros((model.row, 1), dtype=np.float64)
        s = np.concatenate((z, y), axis=1)
        model.dist = s

    def di(model):
        for i in range(model.row):
            model.dist[i, 1] = np.sum(np.square(model.x[i, :] - model.y))

    def data_sort(model):
        model.dist = model.dist[model.dist[:, 1].argsort()]

    def out(model, k):
        model.di()
        model.data_sort()
        ans = []
        for i in range(int(k)):
            ind = int(model.dist[i, 0])
            ans.append(ind)
        return ans
