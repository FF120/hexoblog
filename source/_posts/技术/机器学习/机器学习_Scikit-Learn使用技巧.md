---
title: 深度学习_Scikit-Learn机器学习算法的使用
toc: true

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

```python
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

>>>sel
>>>VarianceThreshold(threshold=0.16)

X2 = sel.fit_transform(X)

>>>X2
>>>
array([[0, 1],
       [1, 0],
       [0, 0],
       [1, 1],
       [1, 0],
       [1, 1]])
```
计算每维特征的方差
```python
a1 = np.array([0,0,1,0,0,0])

>>>a1.var()
>>>0.13888888888888892

a2 = np.array([0,1,0,1,1,1])

>>>a2.var()
>>>Out[161]: 0.22222222222222224

a3 = np.array([1,0,0,1,0,1])

>>>a3.var()
>>>Out[163]: 0.25
```

可以看到，方差小于0.16的只有第一维特征，所以X2保留下来的是原来的第二维和第三维特征。
>这应该是最简单的特征选择方法了：假设某特征的特征值只有0和1，并且在所有输入样本中，95%的实例的该特征取值都是1，那就可以认为这个特征作用不大。如果100%都是1，那这个特征就没意义了。当特征值都是离散型变量的时候这种方法才能用，如果是连续型变量，就需要将连续变量离散化之后才能用，而且实际当中，一般不太会有95%以上都取某个值的特征存在，所以这种方法虽然简单但是不太好用。可以把它作为特征选择的预处理，先去掉那些取值变化小的特征，然后再从接下来提到的的特征选择方法中选择合适的进行进一步的特征选择。

#### Univariate feature selection （单变量特征选择）
主要使用统计的方法计算各个统计值，再根据一定的阈值筛选出符合要求的特征，去掉不符合要求的特征。
#### 主要的统计方法
- F值分类 `f_classif`
- F值回归 `f_regression`
- 卡方统计 `chi2` (适用于非负特征值 和 稀疏特征值)

#### 主要的选择策略
- 选择排名前K的特征 `SelectKbest`
- 选择前百分之几的特征  `SelectPercentile`
- `SelectFpr`  Select features based on a false positive rate test.
- `SelectFdr`  Select features based on an estimated false discovery rate.
- `SelectFwe`  Select features based on family-wise error rate.
- `GenericUnivariateSelect` Univariate feature selector with configurable mode.

>`false positive rate`:  FP / (FP + TP) 假设类别为0，1；记0为negative,1为positive, `FPR`就是实际的类别是0，但是分类器错误的预测为1的个数 与 分类器预测的类别为1的样本的总数（包括正确的预测为1和错误的预测为1） 的比值。
>`estimated false discovery rate`: 错误的拒绝原假设的概率
>`family-wise error rate`: 至少有一个检验犯第一类错误的概率


假设检验的两类错误：
> - 第一类错误：原假设是正确的，但是却被拒绝了。(用α表示）
> - 第二类错误：原假设是错误的，但是却被接受了。(用β表示)

#### 具体应用
```python

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
#SelectKBest -- f_classif
from sklearn.feature_selection import f_classif
iris = load_iris()
X, y = iris.data, iris.target
X_fitted = SelectKBest(f_classif, k=3).fit(X,y)
print "SelectKBest -- f_classif"
print X_fitted.scores_
print X_fitted.pvalues_
print X_fitted.get_support()
X_transformed = X_fitted.transform(X)
print X_transformed.shape
#SelectKBest -- chi2
from sklearn.feature_selection import chi2
X_fitted_2 = SelectKBest(chi2, k=3).fit(X,y)
print "SelectKBest -- chi2"
print X_fitted_2.scores_
print X_fitted_2.pvalues_
print X_fitted_2.get_support()
X_transformed_2 = X_fitted_2.transform(X)
print X_transformed_2.shape

#SelectPercentile -- f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
X_fitted_3 = SelectPercentile(f_classif, percentile=50).fit(X,y)
print "SelectPercentile -- f_classif"
print X_fitted_3.scores_
print X_fitted_3.pvalues_
print X_fitted_3.get_support()
X_transformed_3 = X_fitted_3.transform(X)
print X_transformed_3.shape

#SelectPercentile -- chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
X_fitted_4 = SelectPercentile(chi2, percentile=50).fit(X,y)
print "SelectPercentile -- chi2"
print X_fitted_4.scores_
print X_fitted_4.pvalues_
print X_fitted_4.get_support()
X_transformed_4 = X_fitted_4.transform(X)
print X_transformed_4.shape

#SelectFpr --- chi2
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
X_fitted_5 = SelectFpr(chi2, alpha=2.50017968e-15).fit(X,y)
print "SelectFpr --- chi2"
print X_fitted_5.scores_
print X_fitted_5.pvalues_
print X_fitted_5.get_support()
X_transformed_5 = X_fitted_5.transform(X)
print X_transformed_5.shape

#SelectFpr --- f_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
X_fitted_6 = SelectFpr(f_classif, alpha=1.66966919e-31 ).fit(X,y)
print "SelectFpr --- f_classif"
print X_fitted_6.scores_
print X_fitted_6.pvalues_
print X_fitted_6.get_support()
X_transformed_6 = X_fitted_6.transform(X)
print X_transformed_6.shape

# SelectFdr  和 SelectFwe 的用法和上面类似，只是选择特征时候的依据不同，真正决定得分不同的是
#统计检验方法，从上面可以看到，使用f_classif的得出的参数都相同。

>>>
SelectKBest -- f_classif
[  119.26450218    47.3644614   1179.0343277    959.32440573]
[  1.66966919e-31   1.32791652e-16   3.05197580e-91   4.37695696e-85]
[ True False  True  True]
(150L, 3L)
SelectKBest -- chi2
[  10.81782088    3.59449902  116.16984746   67.24482759]
[  4.47651499e-03   1.65754167e-01   5.94344354e-26   2.50017968e-15]
[ True False  True  True]
(150L, 3L)
SelectPercentile -- f_classif
[  119.26450218    47.3644614   1179.0343277    959.32440573]
[  1.66966919e-31   1.32791652e-16   3.05197580e-91   4.37695696e-85]
[False False  True  True]
(150L, 2L)
SelectPercentile -- chi2
[  10.81782088    3.59449902  116.16984746   67.24482759]
[  4.47651499e-03   1.65754167e-01   5.94344354e-26   2.50017968e-15]
[False False  True  True]
(150L, 2L)
SelectFpr --- chi2
[  10.81782088    3.59449902  116.16984746   67.24482759]
[  4.47651499e-03   1.65754167e-01   5.94344354e-26   2.50017968e-15]
[False False  True False]
(150L, 1L)
SelectFpr --- f_classif
[  119.26450218    47.3644614   1179.0343277    959.32440573]
[  1.66966919e-31   1.32791652e-16   3.05197580e-91   4.37695696e-85]
[False False  True  True]
(150L, 2L)
```

#### Recursive feature elimination （递归特征消除）
使用某种方法，给每一维特征赋一个权重（例如线性回归的系数），去除系数最小的K个特征，然后在剩下的特征上重复上述方法，直到剩下的特征满足特征选择个数的要求。
```python
'''

用SVM获得每个特征对分类结果的贡献程度，按照贡献程度从大到小排名，选出贡献程度最大的
前K个特征作为特征选择的结果,使用SVM的时候，排名的依据是fit之后的coef_值。

这里的估计器可以替换成任何其他方法，如GLM
'''

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
print "原来的特征："
print X.shape

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
ref = rfe.fit(X, y)
print "选择的特征的个数"
print np.sum(ref._get_support_mask())
print ref._get_support_mask()
print rfe.ranking_

>>>
原来的特征：
(1797L, 64L)
选择的特征的个数
10
[False False False False  True False False False False False False False
 False False False False False False False False False  True False False
 False False  True False False False  True False False False False False
 False False  True False False False  True False False  True  True False
 False False False False False  True False False False False  True False
 False False False False]
[55 41 22 14  1  8 25 42 48 28 21 34  5 23 35 43 45 32 10  6 19  1 30 44 46
 36  1  9 11 29  1 50 54 33 16 26 20  7  1 53 52 31  1  2  4  1  1 49 47 38
 17 27 15  1 13 39 51 40  1 18 24 12  3 37]
```

使用上面的方法，需要人为的确定最后输出的特征的个数，如果不知道需要多少特征才能达到好的效果，可以使用下面的交叉验证方法自动确定输出几个特征最优。
```python
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

#产生人工数据
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=5,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 5),
              scoring='accuracy')
rfecv = rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("选择的特征：")
print rfecv.support_
```

#### Feature selection using SelectFromModel(从模型中选择特征)

许多估计模型在执行完fit方法以后都会有`coef_`参数，这个参数实际上是各个特征的权重，所以我们可以根据这个权重选择特征，把权重小的特征去除。
```python
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()
clf.fit(X,y)
# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold='mean',prefit=True)
print X.shape
#sfm = sfm.fit(X, y)
print "============LassoCV================"
print "选择的特征"
print sfm._get_support_mask();
n_features = sfm.transform(X).shape[1]
print n_features

# We use LinearSVC
from sklearn.svm import LinearSVC
#C 越小，选择的特征越少
lsvc = LinearSVC(C=0.001, penalty="l1", dual=False)
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
lsvc.fit(X,y)
model = SelectFromModel(lsvc, prefit=True)
print "============线性SVM==============================="
print "选择的特征"
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features


from sklearn import linear_model
clf = linear_model.LogisticRegression(C=0.001, penalty='l1', tol=1e-6)
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
clf.fit(X,y)
model = SelectFromModel(clf, prefit=True)
print "============逻辑回归==============================="
print "选择的特征"
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
print "============基于树的特征选择==============================="
print clf.feature_importances_
print "选择的特征："
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features

>>>
(506L, 13L)
============LassoCV================
选择的特征
[False False False False False  True False  True False False  True False
  True]
4
============线性SVM===============================
选择的特征
[False  True False False False False  True False False  True False  True
 False]
4
============逻辑回归===============================
选择的特征
[False False False False False False False False False  True False  True
 False]
2
============基于树的特征选择===============================
[ 0.12196356  0.02193675  0.03935991  0.01633832  0.0721041   0.13938681
  0.11703915  0.10962258  0.03116833  0.04455059  0.04134067  0.1074465
  0.13774273]
选择的特征
[ True False False False False  True  True  True False False False  True
  True]
6
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

#### 支持向量机（SVM）

##### 简介
scikit-learn支持稠密的(dense)和稀疏的（sparse）数据，但是当测试数据是稀疏的时候，训练数据必须也是稀疏的。为了达到最优的性能，建议稠密数据使用`numpy.ndarray`,稀疏数据使用`scipy.sparse.csr_matrix` and `dtype=float64`.

##### 用途

- 分类（classfication）
- 回归 (regression)
- 离群点检测 (outliers detection)

##### 优点

- 当特征维数很高时很有效(effective in high dimensional space)
- 当特征的维数远大于样本的数量的时候依然有效
- 使用的是训练点（training points）的子集进行决策函数（decision function）的计算，所以是内存高效的（memory efficient）
- 多种可供选择的核函数(kernel function)提高了算法的灵活性，核函数是可以根据自己的需要自定义的。

##### 缺点

- 当特征的数量远大于样本的数量的时候，算法的性能会下降（poor performance）
- SVM不直接提供概率估计（probability estimates），而是使用一个五折交叉验证（five-fold cross-validation）,计算复杂性较高，一般不适合海量数据的处理。

##### 使用方法

###### 二分类

```python
#准备数据
X = [[0, 0], [1, 1]]
y = [0, 1]
#引入支持向量机
from sklearn import svm
'''
创建模型,这里有三种方法:
svm.SVC(); svm.NuSVC(); svm.LinearSVC()
'''
clf = svm.SVC()
'''
训练数据，这里X是[n_samples,n_features],y是[n_labels]
'''
clf = clf.fit(X_train, y)
#使用训练好的模型预测
y_predicted = clf.predict(X_test)

#获得训练好的模型的一些参数
>>> # get support vectors
>>> clf.support_vectors_
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> # get indices of support vectors
>>> clf.support_
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_
array([1, 1]...)
#get the params of the svm
>>>clf.coef_
```
>上面是最简单的支持向量机的使用方式，下一步还需要了解可以设置的各个参数是什么意思，如何设置，如何交叉验证，如何选择和函数。

###### 多分类

```python
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
# "one-against-one"
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
'''
one-against-one 就是一对一，假设这四类的名称为a,b,c,d.
则需要训练区分(a,b)(a,c)(a,d)(b,c)(b,d)(c,d)的6种模型，所以
one-against-one这种策略在做多分类问题的时候会生成n*(n-1)/2个模型，每个模型区分其中的两个类。
'''
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
'''
 "one-vs-the-rest" 就是一对余下所有的，假设四类的名称为a,b,c,d;
 则需要训练区分(a,bcd),(b,acd)(c,abd)(d,abc)的4种模型，每个模型区分其中一个类，被除此类之外的所有类当作另外一个类处理。
 这种策略在做多分类问题的时候会生成n个模型。
'''

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

```
> 一些补充说明：`SVC`和`NuSVC`实现了`one-against-one`(`ovo`)方法，` LinearSVC`实现了`one-vs-test`(`ovr`)和另外一个叫做`Crammer and Singer`的实现多分类的方法，
可以通过指定`multi_class='crammer_singer'`来使用它。不多实践证明，在使用` LinearSVC`的进行多分类的时候，优先选择`one-vs-test`(`ovr`)，
因为`one-vs-test`(`ovr`)和`crammer_singer`得到的结果差不多，但是前者的计算时间要短。

#### 模型参数说明

##### LinearSVC
```python
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.LinearSVC()
clf.fit(X,y)
print clf

>>>
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
```

**参数**
- `C`：可选参数，类型`float`,默认为1.0；误差的惩罚参数
- `class_weight`: 类型`dict`,可选参数，默认每个class的权重都是1.用来设置每个class的权重。
- `dual`:默认为`True`,类型`bool`,当`n_samples` > `n_features`时，设置成`False`.
- `fit_intercept`: 可选参数，类型为bool,默认为True. 意思是为模型计算截距（intercept），当数据事先已经是centered的时候，可以设置成False，不计算截距。
- `intercept_scaling`： 可选参数，类型为float,默认为1.意思是截距是否缩放。
- `loss`: 类型string,只能取"hinge" 和 "squared_hinge",默认取"squared_hinge"；定义SVM的损失函数，"hinge"是标准的SVM损失函数，"squared_hinge"是标准损失函数的平方。
- `max_iter`： 类型为int,默认为1000，模型最大的迭代次数。
- `multi_class`：类型string，只能取'ovr' 和 'crammer_singer' (默认值是'ovr')，当计算多分类的时候，指定多分类采取的策略。‘ovr’是将其中一类和剩下所有类二分，默认用这个策略就好。
- `penalty`： 类型string,只能取'l1' or 'l2' (默认值是'l2')，l1使参数稀疏，l2使大部分参数接近为0但是不是0，详细信息参考“机器学习中的范数”
- `random_state`： 只能取int seed, RandomState instance,  None 三个中的一个，默认值是None,指定产生伪随机数的时候使用的种子（seed）
- `tol`：可选参数，类型为float,默认值是1e-4,指定停止时候的允许的误差。
- `verbose`：类型为int,默认值是0，是否开启详细的输出，默认不要开启就好。如果开启，在多线程的时候可能运行不正确。

**属性**
- `coef_`：训练好之后的SVM模型中的参数的取值（就是系数），当是二分类的时候，shape=[n_features],多分类的时候，shape = [n_classes,n_features]
- `intercept_`:截距，二分类的时候shape = [1] ,多分类的时候shape=[n_classes]

##### SVC

```python
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X,y)
print clf

>>>
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

```

**参数**
- `C`=1.0, 可选参数，类型`float`,默认为1.0；误差的惩罚参数
- `cache_size`=200, 定义模型计算时使用的缓存大小，单位MB。
- `class_weight`=None,类型dict,默认为None,可以设置成'balanced'，这样会根据y自动计算每个class的权重。还可以手动设置每个class的权重。
- `coef0`=0.0,可选参数，类型为float,默认为0.0，独立于核函数（kernel function）的参数，只在'poly' and 'sigmoid'的时候有影响。
- `decision_function_shape`=None, 'ovo', 'ovr' or None, default=None
- `degree`=3, 可选参数，类型为int,默认为3，多项式和函数的度，其他类型的和函数自动忽略该参数。
- `gamma`='auto', 可选参数，类型为float,默认为‘auto’,默认取 1/n_features作为gamma的值。
- `kernel`='rbf',可选参数，类型为string，默认值为‘rbf’,定义SVM所使用的核函数，可选择的项如下：
	- linear
	- poly
	- rbf
	- sigmoid
	- precomputed
	- a callable(一个回调函数)
- `max_iter`=-1, 最大迭代次数，默认为-1，意思是无限制。
- `probability`=False, 可选参数，类型bool,默认值为False. 是否进行概率估计，使用之前需要先调用fit方法。
- `random_state`=None, 只能取int seed, RandomState instance,  None 三个中的一个，默认值是None,指定产生伪随机数的时候使用的种子（seed）
- `shrinking`=True,可选参数，类型boolean,默认值为True,是否开启“shrinking heuristic”
- `tol`=0.001, 可选参数，类型为float,默认值是1e-4,指定停止时候的允许的误差。
- `verbose`=False，类型为int,默认值是0，是否开启详细的输出，默认不要开启就好。如果开启，在多线程的时候可能运行不正确。

**属性**
- `support_` : array-like, shape = [n_SV]，支持向量的下标
- `n_support_` : array-like, dtype=int32, shape = [n_class] 每个类的支持向量的个数。
- `support_vectors_` ：shape = [n_SV, n_features]，支持向量(SVM确定了一个分类超平面，支持向量就是平移这个超平面，最先与数据集的交点。)
- `dual_coef_` : array, shape = [n_class-1, n_SV] 在决策函数（decision function）中支持向量的系数
- `coef_` : array, shape = [n_class-1, n_features]，特征的权重，只在线性核的时候可用。
- `intercept_` : array, shape = [n_class * (n_class-1) / 2]，决策函数（decision function）中的常量。

##### NuSVC

```python
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.NuSVC()
clf.fit(X,y)
print clf

>>>
NuSVC(cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
   max_iter=-1, nu=0.5, probability=False, random_state=None,
   shrinking=True, tol=0.001, verbose=False)

```
**参数**
大部分都与`SVC`一样，只是使用了一个额外的参数控制支持向量（support vector）的个数。
- `nu`:可选参数，类型float，默认值是0.5，值必须要(0,1]之间。

**属性**
- `support_` : array-like, shape = [n_SV]，支持向量的下标
- `n_support_` : array-like, dtype=int32, shape = [n_class] 每个类的支持向量的个数。
- `support_vectors_` ：shape = [n_SV, n_features]，支持向量
- `dual_coef_` : array, shape = [n_class-1, n_SV] 在决策函数（decision function）中支持向量的系数
- `coef_` : array, shape = [n_class-1, n_features]，特征的权重，只在线性核的时候可用。
- `intercept_` : array, shape = [n_class * (n_class-1) / 2]，决策函数（decision function）中的常量。

#### 查看训练好的模型的参数

#### 决策函数（decision function）

#### 核函数（kernel function）

优先使用‘rbf’调节参数，当特征的数量远远大于样本的数量的时候，考虑使用线性核函数。

---------------------------------
#### 随机梯度下降（Stochastic Gradient Descen）
分类，回归
##### 简介
随机梯度下降法适用于特征数据大于10的5次方，样本数量大于10的5次方的大规模数据的处理领域。

##### 用途
可以处理大规模数据和稀疏数据。

##### 优点
- 高效
- 易于实现

##### 缺点
- 需要很多超参数
- 对特征的缩放敏感

##### 使用方法

```python
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier()
clf.fit(X, y) #训练
clf.predict([[2., 2.]])  #预测

print clf
>>>
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

>>>clf.coef_  #模型系数
>>>Out[31]: array([[ 9.91080278,  9.91080278]])

>>>clf.intercept_    #截距
>>>array([-9.99002993])
```

**参数**
- `alpha`=0.0001,
- `average`=False,
- `class_weight`=None, epsilon=0.1,
- `eta0`=0.0,
- `fit_intercept`=True,
- `l1_ratio`=0.15,
- `learning_rate`='optimal',
- `loss`='hinge',
- `n_iter`=5, n_jobs=1,
- `penalty`='l2',
- `power_t`=0.5,
- `random_state`=None,
- `shuffle`=True,
- `verbose`=0,
- `warm_start`=False

**属性**
- coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,n_features);Weights assigned to the features.

- intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,);Constants in decision function.

#### 最近邻方法（Nearest Neighbors）

如果一个样本在特征空间中的k个最相 似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别.

##### 简介

scikit-learn实现了监督的和非监督的最近邻方法，决定最近邻的算法有`ball_tree`,`kd_tree`,`brute`,可以通过指定模型参数`algorithm`的值来指定到底使用哪一个算法。
主要功能是实现***分类***和***回归***。

##### 用法

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
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
