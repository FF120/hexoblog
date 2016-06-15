---
title: scikit-learn feature selection
toc: true
categories:
  - 机器学习
tags:
  - scikit-learn
date: 2016-06-14 20:08:11
---
# Sample Baseline approach (基础方法)
特征选择时一个最简单的想法就是 去掉那些对类别区分度不大的特征，例如，所有样本的第一维特征的值都是0，那么这个第一维特征就可以去掉，因为它对区分不同的sample贡献很小。
`scikit-learn`提供一个方法`VarianceThreshold`, 实现这样一个简单的选择特征的策略：去掉那些方差达不到指定阈值的特征。
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

# Univariate feature selection （单变量特征选择）
主要使用统计的方法计算各个统计值，再根据一定的阈值筛选出符合要求的特征，去掉不符合要求的特征。
## 主要的统计方法
- F值分类 `f_classif`
- F值回归 `f_regression`
- 卡方统计 `chi2` (适用于非负特征值 和 稀疏特征值)

## 主要的选择策略
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

## 具体应用
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

# Recursive feature elimination （递归特征消除）
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

# Feature selection using SelectFromModel(从模型中选择特征)
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

#  特征选择方法的比较
