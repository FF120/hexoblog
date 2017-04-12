---
title: 算法专题_C++语言技巧
toc: true
categories:
  - 算法
tags:
  - ACM
date: 2017-04-07 11:29:34
---
## 自定义set的比较函数
存入set的元素默认是有序的，但是默认的比较可能不能满足我们的要求，这个时候
就需要自定义比较的函数。 set的排序是使用红黑树的结构，插入删除和取出最小的
元素都比较高效。
```C++
struct NumBit{
    int num;
    NumBit(int n) : num(n) {}
    bool operator<(const struct NumBit & right)const   //重载<运算符
    {
        vector<int> vtmp1;
        int n = this->num;
        int b = 0;
        while(n){
            b = n % 10;
            vtmp1.insert(vtmp1.begin(),b);
            n /= 10;
        }
        vector<int> vtmp2;
        int n2 = right.num;
        int b2 = 0;
        while(n2){
            b2 = n2 % 10;
            vtmp2.insert(vtmp2.begin(),b2);
            n2 /= 10;
        }
        int i = 0;
        int j = 0;
        int ilen = vtmp1.size();
        int jlen = vtmp2.size();
        while( i<ilen || j<jlen ){
            if(i<ilen && j<jlen && vtmp1[i] > vtmp2[j]){
                return false;
            }else if(i<ilen && j<jlen && vtmp1[i] < vtmp2[j]){
                return true;
            }else if(i<ilen && j<jlen && vtmp1[i] == vtmp2[j]){
                i++;
                j++;
            }else if(i==ilen){
                if(vtmp2[j] > vtmp2[0]) return true;
                else if(vtmp2[j] < vtmp2[0]) return false;
                else if(j == jlen){
                    return false;
                }else{
                    j++;
                }
            }else if(j==jlen){
                if(vtmp1[i] > vtmp1[0]) return false;
                else if(vtmp1[i] < vtmp1[0]) return  true;
                else if(i == ilen){
                    return true;
                }else{
                    i++;
                }
            }else{
                break;
            }
        }
        return false;
    }
};


```

使用的时候直接使用上面定义的结构体作为set的类型
```C++
multiset<NumBit> s; // 
```

## 整数转换成字符串
```C++
#include <sstream>
#include <string>
string Int_to_String(int n)
{
    ostringstream stream;
    stream<<n;  //n为int类型
    return stream.str();
}
```

## 十进制数字转换成K进制
deque<int> Kin(int n,int k){
    deque<int> result;
    while(n/k != 0){
        result.push_front(n%k);
        n = n / k;
    }
    result.push_front(n);
    return result;
}

## K进制数字转换成十进制
```C++
/**
 * 将K进制的deque转换成10进制
 * @param v
 * @return
 */
int Kinverse(deque<int> v,int k){
    int s = 0;
    int i = 0;
    while(!v.empty()){
        s += v.back() * std::pow(float(k),i);
        ++i;
    }
    return s;
}
```