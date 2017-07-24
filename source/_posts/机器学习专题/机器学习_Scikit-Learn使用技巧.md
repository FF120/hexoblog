---
title: scikit-learn机器学习算法的使用
toc: true
categories:
  - 机器学习
tags:
  - scikit-learn
  - python
date: 2017-05-14 18:51:20
---

`scikit-learn`是一个很受欢迎的机器学习方面的`python`工具包，它定义的一些范式和处理流程影响深远，所以，认识和了解一些这个工具包对于自己实现一些机器学习算法是很有帮助的。它已经实现了很多方法帮助我们便捷的处理数据，例如，划分数据集为训练集和验证集，交叉验证，数据预处理，归一化等等。

<!-- more -->

### 预测结果与真实结果的比较

```python
# 计算均方误差
from sklearn import metrics
rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))

# 计算准确率
acc = metrics.accuracy_score(y_test, y_pred)

# 混淆矩阵
cm = metrics.confusion_matrix(y_test, y_pred)

# classification_report
cr = metrics.classification_report(y_true, y_pred)

# ROC AUC曲线
from sklearn.metrics import roc_curve, auc

```

### 划分数据集

```python
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3, random_state=0)

# 分折
from sklearn.cross_validation import KFold
kf = KFold(n_samples, n_folds=2)
for train, test in kf:
    print("%s %s" % (train, test))

# 保证不同的类别之间的均衡，这里需要用到标签labels
from sklearn.cross_validation import StratifiedKFold
labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(labels, 3)
for train, test in skf:
    print("%s %s" % (train, test))

# 留一交叉验证
from sklearn.cross_validation import LeaveOneOut
loo = LeaveOneOut(n_samples)
for train, test in loo:
    print("%s %s" % (train, test))

# 留P交叉验证
from sklearn.cross_validation import LeavePOut
lpo = LeavePOut(n_samples, p=2)
for train, test in lpo:
    print("%s %s" % (train, test))

# 按照额外提供的标签留一交叉验证,常用的情况是按照时间序列
from sklearn.cross_validation import LeaveOneLabelOut
labels = [1, 1,1, 2, 2]
lolo = LeaveOneLabelOut(labels)
for train, test in lolo:
    print("%s %s" % (train, test))

# 按照额外提供的标签留P交叉验证
from sklearn.cross_validation import LeavePLabelOut
labels = [1, 1, 2, 2, 3, 3,3]
lplo = LeavePLabelOut(labels, p=2)
for train, test in lplo:
    print("%s %s" % (train, test))

# 随机分组
from sklearn.cross_validation import ShuffleSplit
ss = ShuffleSplit(16, n_iter=3, test_size=0.25,random_state=0)
for train_index, test_index in ss:
    print("%s %s" % (train_index, test_index))

# 考虑类别均衡的随机分组
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
for train, test in sss:
    print("%s %s" % (train, test))
```

### 特征选择方法
```python
# 去除方差较小的特征
from sklearn import feature_selection
vt = feature_selection.VarianceThreshold(threshold='')
vt.fit(X_train)
X_train_transformed = vt.transform(X_train)
X_test_transformed = vt.transform(X_test)

# 按照某种排序规则 选择前K个特征
# 除了使用系统定义好的函数f_classif，还可以自己定义函数
sk = SelectKBest(feature_selection.f_classif,k=100)
sk.fit(X_train,y_train)
X_train_transformed = sk.transform(X_train)
X_test_transformed = sk.transform(X_test)

# 递归特征消除
rfecv = RFECV(estimator=svc, step=step, cv=StratifiedKFold(y, n_folds = n_folds),scoring='accuracy')
rfecv.fit(X_train, y_train)
X_train_transformed = rfecv.transform(X_train)
X_test_transformed = rfecv.transform(y_train)

# 使用L1做特征选择
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=1, penalty="l1", dual=False)
lsvc.fit(X_train,y_train)
X_train_transformed = lsvc.transform(X_train)
X_test_transformed = lsvc.transform(y_train)

# 基于树的特征选择
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)
X_train_transformed = etc.transform(X_train)
X_test_transformed = etc.transform(X_test)

# 基于线性判别分析做特征选择
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
lda.fit(X_train, y_train)
X_train_transformed = lda.transform(X_train)
X_test_transformed = lda.transform(X_test)
```

### 分类器

```python
# linear_model
from sklearn import linear_model

lmlr = linear_model.LinearRegression()
lmlr.fit(X_train,y_train)
lmlr.coef_
predicted_y = lmlr.predict(X_test)

# L1 惩罚项
lmr = linear_model.Ridge (alpha = .5)
lmr.fit(X_train,y_train)
lmr.coef_
lmr.intercept_
predicted_y = lmr.predict(X_test)

lmrcv = linear_model.RidgeCV(alphas=[0.1, 0.5,1.0, 10.0]) # 自带交叉验证
lmrcv.fit(X_train,y_train)
lmrcv.alpha_
predicted_y = lmrcv.predict(X_test)

# L2 惩罚项
lmla = linear_model.Lasso(alpha = 0.001)
lmla.fit(X_train,y_train)
predicted_y = lmla.predict(X_test)

# L1 + L2 惩罚项的一个混合
lmela = linear_model.ElasticNet(alpha=0.01,l1_ratio=0.9)
lmela.fit(X_train,y_train)
predicted_y = lmela.predict(X_test)

"""
Least Angle Regression : 适用于高维数据，缺点是对噪声比较敏感
"""
lmlar = linear_model.Lars(n_nonzero_coefs=10)
lmlar.fit(X_train,y_train)
predicted_y = lmlar.predict(X_test)

"""
BayesianRidge : Bayesian Ridge Regression
小特征数目表现不佳
"""
lmbr = linear_model.BayesianRidge()
lmbr.fit(X_train,y_train)
lmbr.coef_
predicted_y = lmbr.predict(X_test)

"""
ARDRegression : similar to BayesianRidge, but tend to sparse
"""
lmardr = linear_model.ARDRegression(compute_score=True)
lmardr.fit(X_train, y_train)
predicted_y = lmardr.predict(X_test)

"""
逻辑回归
Logistic regression
"""
lmlr1 = linear_model.LogisticRegression(C=1, penalty='l1', tol=0.01)
lmlr2 = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.01)
lmlr1.fit(X_train,y_train)
predicted_y = lmlr1.predict(X_test)

"""
SGDClassifier
"""
lmsdg = linear_model.SGDClassifier()
lmsdg.fit(X_train,y_train)
predicted_y = lmsdg.predict(X_test)

"""
Perceptron : 感知机算法
"""
lmper = linear_model.Perceptron()
lmper.fit(X_train,y_train)
predicted_y = lmper.predict(X_test)

"""
PassiveAggressiveClassifier : similar to Perceptron but have peny
"""
lmpac = linear_model.PassiveAggressiveClassifier()
lmpac.fit(X_train,y_test)
predicted_y = lmpac.predict(X_test)

"""
Linear discriminant analysis  && quadratic discriminant analysis
"""
from sklearn.lda import LDA
lda = LDA(solver="svd", store_covariance=True)
lda.fit(X, y)
predicted_y = lda.predict(X_test)

from sklearn.qda import QDA
qda = QDA()
qda.fit(X, y, store_covariances=True)
predicted_y = qda.predict(X_test)

"""
Kernel ridge regression:
combines Ridge Regression (linear least squares with l2-norm regularization)
with the kernel trick
"""
from sklearn.kernel_ridge import KernelRidge
kr = KernelRidge(alpha=0.1)
kr.fit(X,y)
predicted_y = kr.predict(X_test)

"""
Support Vector Machines : 支持向量机分类
"""
from sklearn import svm
svmsvc = svm.SVC(C=0.1,kernel='rbf')
svmsvc.fit(X_train,y_train)
svmsvc.score(X_test,y_test)

"""
Support Vector Regression.
"""
svmsvr = svm.SVR()
svmsvr.fit(X_train,y_train)
svmsvr.score(X_test,y_test)

"""
Nearest Neighbors : 最近邻
"""
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

from sklearn.neighbors import KNeighborsClassifier
nkc = KNeighborsClassifier(15, weights='uniform')
nkc.fit(X_train,y_train)
nkc.score(X_test,y_test)

from sklearn.neighbors import NearestCentroid
clf = NearestCentroid(shrink_threshold=0.1)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
```

### 模型持久化

```python
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01)
joblib.dump(clf_l1_LR, 'LogisticRegression.model')
```

### 结果的可视化

```python
import matplotlib.pyplot as plt

plt.figure()
plt.title("VarianceThreshold For Feature Selection")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(lsvc_feature_num, lsvc_score)
plt.show()
```

### 数据预处理

`scikit-learn`提供了很多数据预处理的方法，使用的时候需要引入的包是`preprocessing`.

**缩放scale**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 根据最大值和最小值缩放到[0,1]范围
min_max_scaler = MinMaxScaler()
X_transformed = min_max_scaler.fit_transform(X)

# 数据标准化，使得均值为0，方差为1
ss = StandardScaler()
X_transformed = ss.fit_transform(X)

# 考虑离群点的缩放，首先排除离群点再缩放
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_transformed = robust_scaler.fit_transform(X)
```

**one-hot编码**

对于离散的类别特征，可以使用`one-hot`编码来处理特征，这样处理之后的特征可以直接被一些学习器使用。该方法默认会根据类别的数量生成能够表示该类别的二进制编码。

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
transformed_data = enc.transform(data).toarray()
```

**特征组合**

特征组合的一个最简单的尝试是生成多项式特征，例如，如果有两个特征x_1,x_2,多项式为2的特征会自动生成1, x1,x2,x1*x2,x1^2,x2^2 这些特征。

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
X_transformed = poly.fit_transform(X)  
```
