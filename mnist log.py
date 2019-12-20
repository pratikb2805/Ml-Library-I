import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import csv
wt=pd.read_csv('mnist_wts.csv',header=None)
filename=input("Give the name or path of image enclosed in ' ' " )
file=plt.imread(filename)

