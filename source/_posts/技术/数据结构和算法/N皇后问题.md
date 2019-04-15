---
title: N皇后问题
toc: true

tags:
  - ACM
date: 2017-07-20 14:58:58
---

在一个N*N的棋盘上放置N个皇后，要求任意两个皇后不能在同一行、同一列和同一对角线上，求有多少种放置的方法。

<!--more-->

该问题的基本思路是搜索所有的状态。先在第一行的第一个位置放置皇后，然后在第二行找一个合法的位置，接着在第三行找一个合法的位置，一旦达到最后一行，就产生了一个解，如果到某一行的时候没有解了，就回溯到上一行，移动刚才放置的皇后到下一个可行的位置，继续向下搜索。

```c
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <deque>
#include <stack>
using namespace std;

/**
 * 判断在棋盘的row,col位置是否可以放置皇后
 * @param C  c[i]=j 表示第i行第j列有皇后
 * @param row
 * @param col
 * @return
 */
bool isValid(const vector<int> &C,int &row,int &col){
    for (int i = 0; i < row; ++i) {
        // 在col列已经有皇后
        if(C[i] == col) return false;
        // 对角线上的皇后行坐标相减总是等于列坐标
        if(abs(i-row) == abs(C[i]-col)) return false;
    }
    return true;
}

/**
 *
 * @param C
 * @param row
 */
void dfs(vector<int> &C,int row,int &sum){
    const int N = C.size();
    if(row == N){
        sum++;
        return;
    }
    //测试每一列
    for (int j = 0; j < N; ++j) {
        if(isValid(C,row,j)){
            C[row] = j; //(row,j) 放置一个皇后
            dfs(C,row+1,sum);
        }else{
            continue; // 不行就尝试下一个位置
        }
    }
}
//N皇后问题
int totalNQueens(int n) {
    vector<int> C(n,-1);
    int sum = 0;
    dfs(C,0,sum);
    return sum;
}


int main(){
    cout<<totalNQueens(4)<<endl;
    return 0;
}
```

由于放置和不放置皇后只需要一个标识，所以可以使用位来标记，这样可以大大节省空间，也可以使用位运算来加速算法。

```c

```
