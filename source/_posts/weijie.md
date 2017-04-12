---
title: weijie
toc: true
categories:
  - 算法
tags:
  - ACM
date: 2017-03-30 18:27:02
---

## 没有解决的问题

### 求数组的单调和
![20170330182803.png](20170330182803.png)

### 二分图的问题

### 霍夫曼编码的实现

### C++ 生成全排列的算法
```C++
#include <iostream>
#include <algorithm>
using namespace std;
void print(vector<int> v){
    for(auto i : v){
        cout<<i<<" ";
    }
    cout<<endl;
}
int main() {
    vector<int> v={4,3,2,1,0};
    print(v);
    //next_permutation(v.begin(),v.end())的作用是生成v的下一个全排列
    while(next_permutation(v.begin(),v.end())){
        print(v);
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```
