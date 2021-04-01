import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

# 初始化 w 和 b，np.array 相当于定义向量
w, b = np.array([0, 0]), 0


# 定义d(x)函数
def d(x):
    return np.dot(w, x) + b  # np.dot是向量的点积


# 历史信用卡发行数据
# 这里的数据集不能随便修改，否则下面的暴力实现可能停不下来
x = np.array([[5, 2], [3, 2], [2, 7], [1, 4], [6, 1], [4, 5]])
y = np.array([-1, -1, 1, 1, -1, 1])


# 感知机的暴力实现
is_modified = True # 记录是否有分错的点

