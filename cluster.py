import numpy as np
import matplotlib.pyplot as plt
from k_means import *
data= np.random.randn(100, 2)*10
model= K_Means(data, 5)
model.train(100)

colors=["red","green","blue", "orange", "yellow"]
model.load()

