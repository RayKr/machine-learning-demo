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
X = np.array([[5, 2], [3, 2], [2, 7], [1, 4], [6, 1], [4, 5]])
y = np.array([-1, -1, 1, 1, -1, 1])

# 感知机的暴力实现
is_modified = True  # 记录是否有分错的点
while is_modified:  # 循环，直到没有分错的点
    is_modified = False

    # 顺序遍及数据集 X
    for xi, yi in zip(X, y):
        # 如果有分错的
        if yi * d(xi) <= 0:
            # 更新法向量 w 和 b
            w, b = w + yi * xi, b + yi
            is_modified = True
            break


# 下面是绘制的代码，主要展示暴力实现的结果，看不懂也没有关系
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 训练 skrlearn 中的感知机，这里是为了借用该感知机的接口，便于绘制决策区域
clf = Perceptron().fit(X, y)
# 根据上面暴力实现得到的 w 和 b 来修改感知机
clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0] = w[0], w[1], b

# 设置字体大小
plt.rcParams.update({'font.size': 14})
# 设置画布和坐标系
fig, ax = plt.subplots(figsize=(6, 3), nrows=1, ncols=1)
fig.subplots_adjust(left=0.25, right=0.75, top=0.999, bottom=0.001)
ax.set_xticks(()), ax.set_yticks(())

cm = ListedColormap(('blue', 'red'))
markers = ('x', 'o')

# 决定绘制区域的大小
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

# 绘制决策区域
plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.4)

# 绘制决策直线
lx = np.linspace(xx.min(), xx.max())
ly = - w[0] / w[1] * lx - b / w[1]
ax.plot(lx, ly, 'k-')

# 根据类别不同，绘制不同形状的点
vmin, vmax = min(y), max(y)
for cl, m in zip(np.unique(y), markers):
    ax.scatter(x=X0[y == cl], y=X1[y == cl], c=y[y == cl], alpha=1, vmin=vmin, vmax=vmax, cmap=cm, edgecolors='k',
               marker=m)

plt.show()

