---
title: 算法专题_hihocoder
toc: true

tags:
  - ACM
date: 2017-08-13 20:19:33
---

hihocoder 每周编程比赛

<!--more-->

## [[Offer收割]编程练习赛22](http://hihocoder.com/contest/offers22)

### [顺序三元组](http://hihocoder.com/problemset/problem/1550)

给定一个长度为N的数组A=[A1, A2, ... AN]，已知其中每个元素Ai的值都只可能是1, 2或者3。 请求出有多少下标三元组(i, j, k)满足1 ≤ i < j < k ≤ N且Ai < Aj < Ak。  

分析：统计2前面有多少个1，2前面有多少个12就可以了。

```c
#include <bits/stdc++.h>

using namespace std;
using ll = long long;

ll solve(vector<int> &v,int n){
    if(v.empty() || n <= 0) return 0;
    ll c1 = 0, c12 = 0, c123 = 0;
    for (int i = 0; i < n; ++i) {
        if(v[i] == 1){
            c1++;
        }else if(v[i] == 2){
            c12 = c12 + c1;
        }else if(v[i] == 3){
            c123 = c123 + c12;
        }else{
            // 输入数据有错
        }
    }
    return c123;
}

int main(){
    freopen("d:/A.in","r",stdin);
    int n;
    cin >> n;
    vector<int> v(n,0);
    for (int i = 0; i < n; ++i) {
        cin>>v[i];
    }
    ll result = solve(v,n);
    cout<<result<<endl;
    return 0;
}
```

### [合并子目录](http://hihocoder.com/problemset/problem/1551)

小Hi的电脑的文件系统中一共有N个文件，例如：

/hihocoder/offer22/solutions/p1

/hihocoder/challenge30/p1/test  

/game/moba/dota2/uninstall  

小Hi想统计其中一共有多少个不同的子目录。上例中一共有8个不同的子目录：

/hihocoder

/hihocoder/offer22

/hihocoder/offer22/solutions

/hihocoder/challenge30

/hihocoder/challenge30/p1

/game

/game/moba

/game/moba/dota2/

前缀树

```c
#include <bits/stdc++.h>
using namespace std;
int N;
int ret = 0;
struct FS {
    string name;
    unordered_map<string, FS*> subs;
    FS(const string &n): name(n){};
};

vector<string> parse(const string &path) {
    vector<string> ret;
    // 需要的字符串前后都有/,first_表示前面/的位置，second_表示后面.的位置
    int first_ = 0;
    int second_ = 0;
    while(first_ < path.size()) {
        second_ = path.find('/', first_);
        // 没有找到/,退出循环
        if (second_ == -1) {
            break;
        }

        if (second_ > first_) {
            ret.push_back(string(path.begin() + first_, path.begin() + second_));
        }

        first_ = second_ + 1;
    }

    return ret;
}

void dfs(FS *root) {
    if (!root) return;
    ret++;
    for(auto p: root->subs) {
        dfs(p.second);
    }
}
int main () {
    freopen("d:/A.in","r",stdin);
    cin >> N;
    FS* root= new FS("");
    string path;
    for(int i = 0; i < N; ++i) {
        cin >> path;
        vector<string> dirs = parse(path);

        // 构造前缀树
        auto cur = root;
        for(auto dir: dirs) {
            if (cur->subs.count(dir) == 0) {
                FS *ndir = new FS(dir);
                cur->subs[dir] = ndir;
            }
            cur = cur->subs[dir];
        }

    }

    // 遍历树，统计结果
    dfs(root);
    cout << ret - 1 << endl;
    return 0;
}
```

### [缺失的拼图](http://hihocoder.com/problemset/problem/1552)

小Hi在玩一个拼图游戏。如下图所示，整个拼图是由N块小矩形组成的大矩形。现在小Hi发现其中一块小矩形不见了。给定大矩形以及N-1个小矩形的顶点坐标，你能找出缺失的那块小矩形的顶点坐标吗？

分析： 每个矩形用四个点的坐标来表示，那么如果不缺失矩形，每个点应该出现偶数次。缺失矩形的地方，每个点只出现奇数次，这样就能找到缺失矩形的坐标。

```c
#include <bits/stdc++.h>

using namespace std;

int main(){
    ios::sync_with_stdio(false);
    freopen("d:/A.in","r",stdin);
    int n;
    cin >> n;
    map<pair<int,int>,int> m;
    for (int i = 0; i < n; ++i){
        int x1,y1,x2,y2;
        cin>>x1>>y1>>x2>>y2;
        m[make_pair(x1,y1)]++;
        m[make_pair(x1,y2)]++;
        m[make_pair(x2,y1)]++;
        m[make_pair(x2,y2)]++;
    }
    vector<int> xx;
    vector<int> yy;
    for (auto i : m) {
        if(i.second % 2 == 0) continue;
        xx.push_back(i.first.first);
        yy.push_back(i.first.second);
    }
    sort(xx.begin(),xx.end());
    sort(yy.begin(),yy.end());
    cout<<xx[0]<<" "<<yy[0]<<" "<<xx[3]<<" "<<yy[3]<<endl;
    return 0;
}
```