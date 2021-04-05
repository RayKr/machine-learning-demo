import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap

# 超参数
epochs = 100  # 固定的迭代次数

# 参数
w, b = np.array([0, 0]), 0  # np.array 相当于定义向量


# 定义 d(x) 函数
def d(x):
    return np.dot(w, x) + b  # np.dot 是向量的点积


# 定义 sign 函数
def sign(x):
    return 1 if x >= 0 else -1


# 定义 h(x) 函数
def h(x):
    return sign(d(x))


# 计算决策边界的评分
def clf_score(X, y):
    score = 0
    for xi, yi in zip(X, y):
        score += yi * h(xi)
    return score


# 感知机的口袋算法
def PLA_pocket(X, y):
    global epochs, w, b

    w, b = np.array([0, 0]), 0  # np.array 相当于定义向量
    best_w, best_b = w, b
    best_cs = clf_score(X, y)
    for _ in range(epochs):

        # 顺序遍及数据集 X
        for xi, yi in zip(X, y):
            # 如果有分错的
            if yi * d(xi) <= 0:
                # 更新法向量 lw 和 lb
                w, b = w + yi * xi, b + yi
                # 对新得到的决策边界进行评分
                cs = clf_score(X, y)
                # 如果更好，则进行更新
                if cs > best_cs:
                    best_cs = cs
                    best_w, best_b = w, b
                break

    w, b = best_w, best_b


# 以下是训练代码
# 载入iris数据集
iris = datasets.load_iris()
# 取后面100个数据，并且只取最后两个特征，以及取出对应的类别
sampleNumber = 100
X = iris.data[50:50 + sampleNumber, [2, 3]]
# iris 数据集的类别是0, 1, 2，为了运用我们实现的感知机算法，这里将后两个类别改为-1, 1
y = np.where(iris.target[50:50 + sampleNumber] == 1, -1, 1)

# 借助 train_test_split 进行随机分割，按照 8 : 2  的比例划分为训练验证集、测试集
rs = 42
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
print(r'总共有 {} 个数据，其中训练验证集中有 {} 个数据，测试集中有 {} 个数据。'.format(len(X), len(X_tv), len(X_test)))

# 组合两个超参数，计算各种组合得到的验证集准确率的平均值
k = 10
for epochs in range(100, 500, 100):
    kf = KFold(n_splits=k, random_state=rs, shuffle=True)
    val_accuracy = 0
    for idx, (train, val) in zip(range(k), kf.split(X_tv)):
        X_train, y_train, X_val, y_val = X_tv[train], y_tv[train], X_tv[val], y_tv[val]
        PLA_pocket(X_train, y_train)
        val_accuracy += 1 - (len(X_val) - clf_score(X_val, y_val)) / 2 / len(X_val)
    print(r'epochs = {}，k={}，验证集准确率的平均值为 {:.2%}。'.format(epochs, k, val_accuracy / k))
