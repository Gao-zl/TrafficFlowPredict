# 加载mat文件
from scipy.io import loadmat

m = loadmat("BSrf1000.mat")
print(m)
print(m.keys())
