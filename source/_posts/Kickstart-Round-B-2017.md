---
title: Kickstart Round B 2017
toc: true
categories:
  - 算法
tags:
  - ACM
date: 2017-05-07 16:16:13
---
谷歌codejam:https://codejam.withgoogle.com/codejam/contest/11304486/dashboard.
<!--more-->
![](2017-05-07_162055.png)
小规模测试文件：https://drive.google.com/open?id=0B2aHWGYn_JL-WFhaWVMxaXpxWVk
大规模测试文件：https://drive.google.com/open?id=0B2aHWGYn_JL-X1BoUlUzZFptQzQ

小规模的问题直接可以用暴力的方法求解，利用整数枚举一个集合的所有子集，计算最大值和最小值的差别。
```c
#include <iostream>
#include <algorithm>
#include <vector>
#include <bitset>
#include <cmath>
using namespace std;
void run(){
    int num ;
    cin>>num;
    vector<int> v(num,0);
    for(int i=num-1;i>=0;i--){
        cin>>v[i];
    }
    int sum = 0;
    int i=0;
    for(int j=i+1;j<num;j++){
        sum += (v[i]-v[j])*pow(2.0,j-i-1);
        sum %= 1000000007;
        for(int k=i+1;k<j;k++){
            sum += (v[k]-v[j])*pow(2.0,j-k-1);
            sum %= 1000000007;
        }
    }
    cout<<sum<<endl;
}
int main(){
    freopen("d:/A-small-attempt0.in","r",stdin);
    freopen("d:/A.out","w",stdout);
    int T,cas=0;
    cin>>T;
    while (T--){
        cout<<"Case #"<<++cas<<": ";
        run();
    }
    return 0;
}
```

大规模的问题这样就不行了，可以考虑N^2的方法，每次以一个固定的元素作为结尾，统计有多少个这样的子集，然后计算和。

```c
#include <iostream>
#include <algorithm>
#include <vector>
#include <bitset>
#include <cmath>
using namespace std;


void run(){
    int num ;
    cin>>num;
    vector<int> v(num,0);
    for(int i=num-1;i>=0;i--){
        cin>>v[i];
    }
    // 完成读取数据
    //遍历组合
    int sum = 0;
    int i=0;
    for(int j=i+1;j<num;j++){
        sum += (v[i]-v[j])*pow(2.0,j-i-1);
        sum %= 1000000007;
        for(int k=i+1;k<j;k++){
            sum += (v[k]-v[j])*pow(2.0,j-k-1);
            sum %= 1000000007;
        }
    }
    cout<<sum<<endl;
}
int main(){
    freopen("d:/A-large.in","r",stdin);
    freopen("d:/A-large.out","w",stdout);
    int T,cas=0;
    cin>>T;
    while (T--){
        cout<<"Case #"<<++cas<<": ";
        run();
    }
    return 0;
}
```
