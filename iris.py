# coding:utf-8
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# 载入iris数据集
iris = datasets.load_iris()
# 取后面100个数据，并且只取最后两个特征，以及取出对应的类别
sampleNumber = 100
X = iris.data[50:50 + sampleNumber, [2, 3]]
# iris 数据集的类别是0, 1, 2，为了运用我们实现的感知机算法，这里将后两个类别改为-1, 1
y = np.where(iris.target[50:50 + sampleNumber] == 1, -1, 1)

# 借助 train_test_split 进行随机分割，按照 6 : 2 : 2 的比例划分为三种数据集
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42)

# 以下是绘制代码，看不懂没有关系
plt.rcParams.update({'font.size': 18})  # 设置字体大小
# 创建并设置排的两个subfigure
fig, axes = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
plt.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.1, wspace=0.04)

# 在两个并排的subfigure中绘制训练集和测试集
cmaps = (
ListedColormap(('blue', 'red')), ListedColormap(('dodgerblue', 'bisque')), ListedColormap(('forestgreen', 'peru')))
markers, xlabels = ('x', 'o'), ('训练集', '验证集', '测试集')
Xs, ys = (X_train, X_val, X_test), (y_train, y_val, y_test)
for ax, xlabel, cm, X, y in zip(axes.flat, xlabels, cmaps, Xs, ys):
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(xlabel)

    vmin, vmax = min(y), max(y)
    for cl, m in zip(np.unique(y), markers):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=y[y == cl], alpha=1, vmin=vmin, vmax=vmax, cmap=cm,
                   edgecolors='k', marker=m)

plt.show()
