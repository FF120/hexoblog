---
title: 同时使用多个分类器(Scikit-Learn)
toc: true

tags:
  - scikit-learn
date: 2016-06-16 15:11:46
---
scikit-learn里面实现的所有分类器都遵循类似的形式，所以我们使用一个循环语句就可以很方便的应用多种分类器在同一个数据集上。

<!-- more -->

```python 

import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.extmath import density
from sklearn import metrics


categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

data_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True, random_state=42)

categories = data_train.target_names 

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)
#print the size and categoies
print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

def benchmark(clf):
    print('-' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3), "Liblinear model with l2"),
        (LinearSVC(loss='l2', penalty='l1',dual=False, tol=1e-3), "Liblinear model with l1"),
        (SGDClassifier(alpha=.0001, n_iter=50,penalty='l2'), "SGD model with l2"),
        (SGDClassifier(alpha=.0001, n_iter=50,penalty='l1'), "SGD model with l1"),        
        (SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),"SGD model with Elastic-Net penalty" ),       
        (NearestCentroid(), "NearestCentroid without threshold"),
        (MultinomialNB(alpha=.01),"MultinomialNB"),
        (BernoulliNB(alpha=.01),"BernoulliNB"),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),('classification', LinearSVC())]),"LinearSVC with L1-based feature selection"),
        ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
```
运行结果：

```
>>>
2034 documents - 3.980MB (training set)
1353 documents - 2.867MB (test set)
4 categories
()
================================================================================
Ridge Classifier
--------------------------------------------------------------------------------
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
        max_iter=None, normalize=False, random_state=None, solver='sag',
        tol=0.01)
train time: 0.125s
test time:  0.016s
accuracy:   0.897
dimensionality: 33810
density: 1.000000
================================================================================
Perceptron
--------------------------------------------------------------------------------
Training: 
Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
      n_iter=50, n_jobs=1, penalty=None, random_state=0, shuffle=True,
      verbose=0, warm_start=False)
train time: 0.093s
test time:  0.000s
accuracy:   0.885
dimensionality: 33810
density: 0.240158
================================================================================
Passive-Aggressive
--------------------------------------------------------------------------------
Training: 
PassiveAggressiveClassifier(C=1.0, class_weight=None, fit_intercept=True,
              loss='hinge', n_iter=50, n_jobs=1, random_state=None,
              shuffle=True, verbose=0, warm_start=False)
train time: 0.125s
test time:  0.000s
accuracy:   0.902
dimensionality: 33810
density: 0.698994
================================================================================
kNN
--------------------------------------------------------------------------------
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
           weights='uniform')
train time: 0.000s
test time:  0.187s
accuracy:   0.858
================================================================================
Liblinear model with l2
--------------------------------------------------------------------------------
Training: 
LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='l2', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=None, tol=0.001, verbose=0)
train time: 0.171sC:\Anaconda2\lib\site-packages\sklearn\svm\classes.py:197: DeprecationWarning: loss='l2' has been deprecated in favor of loss='squared_hinge' as of 0.16. Backward compatibility for the loss='l2' will be removed in 1.0
  DeprecationWarning)
C:\Anaconda2\lib\site-packages\sklearn\svm\classes.py:197: DeprecationWarning: loss='l2' has been deprecated in favor of loss='squared_hinge' as of 0.16. Backward compatibility for the loss='l2' will be removed in 1.0
  DeprecationWarning)

test time:  0.000s
accuracy:   0.900
dimensionality: 33810
density: 1.000000
================================================================================
Liblinear model with l1
--------------------------------------------------------------------------------
Training: 
LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='l2', max_iter=1000, multi_class='ovr',
     penalty='l1', random_state=None, tol=0.001, verbose=0)
train time: 0.203s
test time:  0.000s
accuracy:   0.873
dimensionality: 33810
density: 0.005553
================================================================================
SGD model with l2
--------------------------------------------------------------------------------
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.094s
test time:  0.000s
accuracy:   0.902
dimensionality: 33810
density: 0.671813
================================================================================
SGD model with l1
--------------------------------------------------------------------------------
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='l1', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.327s
test time:  0.000s
accuracy:   0.883
dimensionality: 33810
density: 0.020475
================================================================================
SGD model with Elastic-Net penalty
--------------------------------------------------------------------------------
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='elasticnet', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
train time: 0.499s
test time:  0.000s
accuracy:   0.899
dimensionality: 33810
density: 0.188191
================================================================================
NearestCentroid without threshold
--------------------------------------------------------------------------------
Training: 
NearestCentroid(metric='euclidean', shrink_threshold=None)
train time: 0.016s
test time:  0.000s
accuracy:   0.855
================================================================================
MultinomialNB
--------------------------------------------------------------------------------
Training: 
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
train time: 0.015s
test time:  0.000s
accuracy:   0.900
dimensionality: 33810
density: 1.000000
================================================================================
BernoulliNB
--------------------------------------------------------------------------------
Training: 
BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)
train time: 0.000s
test time:  0.016s
accuracy:   0.884
dimensionality: 33810
density: 1.000000
================================================================================
Random forest
--------------------------------------------------------------------------------
Training: 
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
train time: 2.730s
test time:  0.094s
accuracy:   0.842
================================================================================
LinearSVC with L1-based feature selection
--------------------------------------------------------------------------------
Training: 
Pipeline(steps=[('feature_selection', SelectFromModel(estimator=LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0),
        prefit=False, thresho...ax_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0))])
train time: 0.218s
test time:  0.016s
accuracy:   0.880
```
![many_classfier](many_classfier.png)

上面是使用稀疏矩阵存放特征的一个例子，当然普通的特征也是可以这样做的，例如，可以将上面的数据换成手写数字识别的数据。
```python
iris = load_iris()
X, y = iris.data, iris.target
X_train = X
X_test = X

y_train = y
y_test = y
```
![many_classfier2](many_classfier2.png)
