---
title: KNN with C++
toc: true

tags:
  - KNN
date: 2017-10-15 22:27:57
---

KNN 算法步骤：

- 存储训练集的特征和标签，确定K
- 预测一个未知的样本的时候，计算该样本到每一个训练集中的样本的距离，取前K个距离的最小值
- 在前K个最小值中选择标签出现次数最大的那个，作为预测结果

```c
class KNN{
public:
    int k = 3;
    int feature_num = 0;
    vector<vector<double>> trainSet;
    vector<int> trainLabel;
    // 特征向量之间的距离，修改这里使用不同的距离度量
    double distance(vector<double> &a, vector<double> &b){
        //  返回两个点的距离
        if(a.size() != b.size()) return -1;
        double sum = 0.0;
        for(int i=0;i<a.size();i++){
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        double dis = sqrt(sum);
        return dis;
    }

    KNN() : k(3) {}
    // 创建类的时候保存训练集的特征和标签数据
    KNN(vector<vector<double>> features,vector<int> labels){
        int samples = features.size();
        if(samples != labels.size()){
            throw new exception;
        }
        this->trainSet = features;
        this->trainLabel = labels;
        if(!features.empty()) feature_num = features[0].size();
    }

    // 预测一个测试样例的标签
    int predict(vector<double> test){
        if(test.size() != this->feature_num) throw new exception();
        // 求test到训练集合中所有点的距离，找出距离最小的K个值
        priority_queue<pair<double,int>,vector<pair<double,int>>,lessThan> maxHeap;
        for(int i=0;i<trainSet.size();i++){
            double dis = this->distance(trainSet[i],test);
            if(maxHeap.size() < this->k){
                maxHeap.push(make_pair(dis,trainLabel[i]));
            }else{
                if(dis < maxHeap.top().first){
                    maxHeap.pop();
                }
            }
        }
        // 统计K个点的类别标签，找到出现次数最多的那个标签
        map<int,int> cc; // 统计每个类别从出现的次数 kye
        while(!maxHeap.empty()){
            if(cc.count(maxHeap.top().second) == 0){
                cc[maxHeap.top().second] = 1;
            }else{
                cc[maxHeap.top().second]++;
            }
            maxHeap.pop();
        }
        int maxV = INT32_MIN;
        int label = -1;
        for(auto iter= cc.begin();iter != cc.end(); iter++){
            if(iter->second > maxV){
                maxV = iter->second;
                label = iter->first;
            }
        }
        return label;
    }
    // 预测一个测试样例的标签
    int predict(vector<double> test, int k){
        this->k = k;
        return this->predict(test);
    }
};
```


Kmeans 算法步骤

- 在数据集中随机选择K个点
- 计算所有的点到K个中心点的距离，距离哪个中心点近，就标记成哪个中心点所属的列别
- 计算每个团的新的中心，
- 计算新的中心和上次的中心的差距
- 如果差距大，就继续循环，否则退出


```c
#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <map>

using namespace std;
const double MINVALUE = 0.01;

struct feature{
    vector<double> v;
    int label; // 属于哪一个聚类
};
class Kmeans{
private:
    // 从[start,end] 产生 count 个随机数
    vector<int> random(int start,int end,int count){
        return {1,3,7};
    }
    double distance(feature &a, feature &b){
        return 0;
    }
public:
    int k = 3;
    int cycle = 1000;
    void cluster(vector<feature> dataset){
         vector<int> random3 = random(0,dataset.size()-1,3);
         vector<feature> centers;
         for(int i=0;i<random3.size();i++){
             dataset[i].label = i;
             centers.push_back(dataset[i]);
         }

         while(cycle--) {
             // 标记所有点的所属聚簇
             for (int i = 0; i < dataset.size(); i++) {
                 double minDistance = INT32_MAX;
                 int label = -1;
                 for (int j = 0; j < random3.size(); j++) {
                     double dis = distance(dataset[i], dataset[random3[j]]);
                     if (dis < minDistance) {
                         minDistance = dis;
                         label = j;
                     }
                 }
                 dataset[i].label = label;
             }
             // 更新每个聚簇的中心
             vector<feature> newCenters(centers);
             for (int j = 0; j < dataset[0].v.size(); j++) {
                 vector<double> sum(k, 0.0);
                 vector<int> cc(k, 0);
                 for (int i = 0; i < dataset.size(); i++) {
                     for (int p = 0; p < k; p++) {
                         if (dataset[i].label == p) {
                             sum[p] += dataset[i].v[j];
                             cc[p]++;
                         }
                     }
                 }
                 for (int i = 0; i < k; i++) {
                     sum[i] = sum[i] / cc[i];
                     newCenters[i].v.push_back(sum[i]);
                 }
             }

             // 计算和上次中心的差距，差距在一定范围内就退出,centers, newcenters 之间的差距
             double dis = 0.0;
             for (int i = 0; i < k; i++) {
                 dis += distance(newCenters[i], centers[i]);
             }
             if (dis < MINVALUE) {
                 return;
             }
         }
    }

    void cluster(vector<feature> dataset,int k){
        this->k = k;
        this->cluster(dataset);
    }
};

```