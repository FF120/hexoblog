---
title: weijie
toc: true
categories:
  - �㷨
tags:
  - ACM
date: 2017-03-30 18:27:02
---

## û�н��������

### ������ĵ�����
![20170330182803.png](20170330182803.png)

### ����ͼ������

### �����������ʵ��

### C++ ����ȫ���е��㷨
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
    //next_permutation(v.begin(),v.end())������������v����һ��ȫ����
    while(next_permutation(v.begin(),v.end())){
        print(v);
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

