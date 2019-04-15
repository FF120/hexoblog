---
title: 算法专题_字典树(Trie树)
toc: true

tags:
  - ACM
date: 2017-08-23 09:58:56
---

一种字符串前缀的匹配技术。

<!-- more-->

## 字典树



## 字典树的应用

### 异或（今日头条2017秋招真题）

[异或（今日头条2017秋招真题）](http://exercise.acmcoder.com/online/online_judge_ques?ques_id=3338&konwledgeId=158)

题目描述
									
给定整数m以及n个数字A1, A2, …, An，将数列A中所有元素两两异或，共能得到n(n-1)/2个结果。请求出这些结果中大于m的有多少个。

								
输入
第一行包含两个整数n, m。
第二行给出n个整数A1, A2, …, An。
样例输入
3 10
6 5 10
输出
输出仅包括一行，即所求的答案。
样例输出
2
时间限制
C/C++语言：1000MS其它语言：3000MS	
内存限制
C/C++语言：65536KB其它语言：589824K

思路：

1. 从最高位开始建立字典树，左子树表示二进制0，右子树表示二进制位1. 每个节点统计在n个数字中对应的二进制位上有多少个对应的0或者1.
2. 查询每个数字a和m，比较a和m对应的位，有如下情况
   2.1 a = 0, m = 0, 这个时候 b=0 , a^b = 0 不能确定谁大，继续查找下一位
   2.2 a = 0, m = 1, 这个时候 b=0 , 肯定有a^b < m, 不满足条件，跳过； b = 1, 继续查找下一位
   2.3 a = 1, m = 0, 这个时候 b=0 , 肯定有a^b > m, 满足条件，直接将对应的count加在结果上， b=1,继续查找下一位
   2.4 a = 1, m = 1, 这个时候 b=0 , 继续查找下一位，b = 1,肯定不满足条件，

3. 最后的结果除以2返回， 因为我们既统计了a与b的异或，也统计了b与a的异或。

```c
#include <iostream>
#include <vector>
using namespace std;

struct TrieTree
{
    int count;
    struct TrieTree* next[2]{NULL,NULL};
    TrieTree():count(1){}
};

TrieTree* buildTrieTree(const vector<int>& array)
{
    TrieTree* trieTree = new TrieTree();
    for(int i=0;i<(int)array.size();++i)
    {
        TrieTree* cur = trieTree;
        for(int j=16;j>=0;--j)
        {
            int digit = (array[i] >> j) & 1;
            if(NULL == cur->next[digit])
                cur->next[digit] = new TrieTree();
            else
                ++(cur->next[digit]->count);
            cur = cur->next[digit];
        }
    }
    return trieTree;
}

long long queryTrieTree(TrieTree*& trieTree, const int a, const int m, const int index)
{
    if(NULL == trieTree)
        return 0;

    TrieTree* cur = trieTree;

    for(int i=index;i>=0;--i)
    {
        int aDigit = (a >> i) & 1;
        int mDigit = (m >> i) & 1;

        if(1==aDigit && 1==mDigit)
        {
            if(NULL == cur->next[0])
                return 0;
            cur = cur->next[0];
        }
        else if(0 == aDigit && 1==mDigit)
        {
            if(NULL == cur->next[1])
                return 0;
            cur = cur->next[1];
        }
        else if(1 == aDigit && 0 == mDigit)
        {
            long long val0 =  (NULL == cur->next[0]) ? 0 : cur->next[0]->count;
            long long val1 =  queryTrieTree(cur->next[1],a,m,i-1);
            return val0+val1;
        }
        else if(0 == aDigit && 0 == mDigit)
        {
            long long val0 =  queryTrieTree(cur->next[0],a,m,i-1);
            long long val1 =  (NULL == cur->next[1]) ? 0 : cur->next[1]->count;
            return val0+val1;
        }
    }
    return 0;
}

long long solve(const vector<int>& array, const int& m)
{
    TrieTree* trieTree = buildTrieTree(array);
    long long result = 0;
    for(int i=0;i<(int)array.size();++i)
    {
        result += queryTrieTree(trieTree,array[i],m,16);
    }
    return result /2;
}

int main()
{
    freopen("d:/A.in","r",stdin);
    int n,m;
    while(cin>>n>>m)
    {
        vector<int> array(n);
        for(int i=0;i<n;++i)
            cin>>array[i];
        cout<< solve(array,m) <<endl;
    }
    return 0;
}
```

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

using namespace std;
using ll = long long;

const int MAXBIT = 17;

struct trieTree{
    ll count = 0;
    trieTree* next[2] = {nullptr, nullptr};
    trieTree() : count(0){}
    trieTree(int c) : count(c){}
};

trieTree* createTree(const vector<int> &arr){
    trieTree* root = new trieTree(1);
    for (int i = 0; i < arr.size(); ++i) {
        int index = MAXBIT; // int 整形最多32位
        trieTree* cur = root;
        while(index >= 0){
            int digit = (arr[i] >> index) & 1; // 从低位到高位第index+1位的值
            if(cur->next[digit] == nullptr){
                cur->next[digit] = new trieTree(1);
            }else{
                cur->next[digit]->count ++;
            }
            cur = cur->next[digit];
            index--;
        }
    }
    return root;
}
/**
 * 查询trie树中有多少个数字满足a^b>m
 * @param root
 * @param a
 * @param m
 * @param index
 * @return
 */
ll queryTree(trieTree* root,int a, int m, int index){
    if(root == nullptr) return 0;
    for (int i = index; i >= 0; --i) {
        int aDigit = (a >> i) & 1;
        int mDigit = (m >> i) & 1;
        if(aDigit == 0 && mDigit == 0){
            ll v0 = 0 , v1 = 0;
            if(root->next[1] != nullptr){
                v0 = root->next[1]->count;
            }
            v1 = queryTree(root->next[0],a,m,i-1);
            return v0 + v1;
        }else if(aDigit == 0 && mDigit == 1){
            if(root->next[1] == nullptr) return 0;
            return queryTree(root->next[1],a,m,i-1);
        }else if(aDigit == 1 && mDigit == 0){
            ll v0 = 0 , v1 = 0;
            if(root->next[0] != nullptr){
                v0 = root->next[0]->count;
            }
            v1 = queryTree(root->next[1],a,m,i-1);
            return v0 + v1;
        }else if(aDigit == 1 && mDigit == 1){
            if(root->next[0] == nullptr) return 0;
            return queryTree(root->next[0],a,m,i-1);
        }else{
            cout<<"error"<<endl;
        }
    }
    return 0;
}
ll solve(vector<int> &v, int m){
    trieTree* root = createTree(v);
    ll result = 0;
    for (int i = 0; i < v.size(); ++i) {
        result += queryTree(root,v[i],m,MAXBIT);
    }
    return result / 2;
}

int main(){
    freopen("d:/A.in","r",stdin);
    int n, m ;
    cin >> n >> m;
    vector<int> v(n,0);
    for (int i = 0; i < n; ++i) {
        cin >> v[i];
    }
    cout<<solve(v,m)<<endl;
    return 0;
}

```

### 统计子目录

[统计子目录](http://hihocoder.com/problemset/problem/1551)

描述
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

输入
第一行包含一个整数N (1 ≤ N ≤ 10000)  

以下N行每行包含一个字符串，代表一个文件的绝对路径。保证路径从根目录"/"开始，并且文件名和目录名只包含小写字母和数字。  

对于80%的数据，N个文件的绝对路径长度之和不超过10000  

对于100%的数据，N个文件的绝对路径长度之和不超过500000

输出
一个整数代表不同子目录的数目。

样例输入
3  
/hihocoder/offer22/solutions/p1   
/hihocoder/challenge30/p1/test  
/game/moba/dota2/uninstall
样例输出
8

思路： 用每个目录的名字建立字典树，根是空字符，然后统计整棵树节点的数目，最后返回节点的数目-1.

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

/*
 * [统计子目录](http://hihocoder.com/problemset/solution/1157194)
 * 
 */
struct trieTree{
    string description;
    vector<trieTree*> sons;
    trieTree() : description(""){}
    trieTree(string &s) : description(s){}
};

/**
 * 层次遍历统计节点数目
 * @param root
 * @return
 */
int countNodes(trieTree* root){
    if(root == nullptr) return 0;
    if(root->sons.empty()) return 1;
    vector<trieTree*> nodes, next;
    nodes.push_back(root);

    int re = 0;
    while(!nodes.empty()){
        for (int i = 0; i < nodes.size(); ++i) {
            next.insert(next.end(),nodes[i]->sons.begin(),nodes[i]->sons.end());
        }
        re += nodes.size();
        nodes = next;
        next.clear();
    }
    return re;
}
/**
 * 递归遍历节点数目
 * @param root
 * @return
 */
int countNode2(trieTree *root){
    if(root == nullptr) return 0;
    if(root->sons.empty()) return 1;
    int re = 1;
    for (int i = 0; i < root->sons.size(); ++i) {
        re += countNode2(root->sons[i]);
    }
    return re;
}
vector<string> splitString(const string &s){
    vector<string> re;
    if(s.empty()) return re;
    size_t index1 = 0;
    size_t index2 = 1;
    while(s.find('/',index2) != -1){
        index2 = s.find('/',index1+1);
        re.push_back(s.substr(index1+1,index2-index1-1));
        index1 = index2;
        index2++;
    }
    return re;
}

int solve(vector<string> &pathes, int n){
    if(pathes.empty() || n <= 0) return 0;
    int result = 0;
    trieTree *root = new trieTree();
    for (int i = 0; i < n; ++i) {
        vector<string> path = splitString(pathes[i]);
        trieTree *cur = root;
        for (int j = 0; j < path.size(); ++j) {
            if(cur->sons.empty()){
                cur->sons.push_back(new trieTree(path[j]));
                result++;
                cur = cur->sons[0];
            }else{
                int index = cur->sons.size();
                for (int k = 0; k < cur->sons.size(); ++k) {
                    if(cur->sons[k]->description == path[j]){
                        index = k;
                        break;
                    }
                }
                if(index == cur->sons.size()){
                    cur->sons.push_back(new trieTree(path[j]));
                    result++;
                    cur = cur->sons[cur->sons.size()-1];
                }else{
                    cur = cur->sons[index];
                }
            }
        }
    }
    int re = countNode2(root) - 1;
    return result;
}

int main(){
    freopen("d:/A.in","r",stdin);
    int n;
    cin>>n;
    vector<string> pathes(n,"");
    for (int i = 0; i < n; ++i) {
        cin >> pathes[i];
    }
    cout<<solve(pathes,n)<<endl;
    return 0;
}
```

### 合并子目录

[合并子目录](http://hihocoder.com/problemset/solution/1157744)

描述
小Hi的电脑的文件系统中一共有N个文件，例如：

/hihocoder/offer23/solutions/p1

/hihocoder/challenge30/p1/test  

/game/moba/dota2/uninstall  

经过统计，小Hi认为他的电脑中子目录实在太多了，于是他决定减少子目录的数量。小Hi发现其中一些子目录只包含另一个子目录，例如/hihocoder/offer22只包含一个子目录solution，/game只包含一个子目录moba，而moba也只包含一个子目录dota2。小Hi决定把这样的子目录合并成一个子目录，并且将被合并的子目录的名字用'-'连起来作为新子目录的名字。合并之后上例的3个文件的路径会变为：

/hihocoder/offer23-solutions/p1

/hihocoder/challenge30-p1/test

/game-moba-dota2/uninstall

输入
第一行包含一个整数N (1 ≤ N ≤ 10000)  

以下N行每行包含一个字符串，代表一个文件的绝对路径。保证路径从根目录"/"开始，并且文件名和目录名只包含小写字母和数字。  

对于80%的数据，N个文件的绝对路径长度之和不超过10000  

对于100%的数据，N个文件的绝对路径长度之和不超过500000

输出
对于输入中的每个文件，输出合并子目录之后该文件的绝对路径。

样例输入
3
/hihocoder/offer23/solutions/p1
/hihocoder/challenge30/p1/test
/game/moba/dota2/uninstall
样例输出
/hihocoder/offer23-solutions/p1
/hihocoder/challenge30-p1/test
/game-moba-dota2/uninstall


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
/*
 * [统计子目录](http://hihocoder.com/problemset/solution/1157194)
 *
 */
struct trieTree{
    string description;
    vector<trieTree*> sons;
    trieTree() : description(""){}
    trieTree(string &s) : description(s){}
};

vector<string> splitString(string &s){
    vector<string> re;
    if(s.empty()) return re;
    s.push_back('/');
    size_t index1 = 0;
    size_t index2 = 1;
    while(s.find('/',index2) != -1){
        index2 = s.find('/',index1+1);
        re.push_back(s.substr(index1+1,index2-index1-1));
        index1 = index2;
        index2++;
    }
    return re;
}

trieTree* createTree(vector<string> &pathes){
    if(pathes.empty()) return 0;
    int result = 0;
    trieTree *root = new trieTree();
    for (int i = 0; i < pathes.size(); ++i) {
        vector<string> path = splitString(pathes[i]);
        trieTree *cur = root;
        for (int j = 0; j < path.size(); ++j) {
            if(cur->sons.empty()){
                cur->sons.push_back(new trieTree(path[j]));
                result++;
                cur = cur->sons[0];
            }else{
                int index = cur->sons.size();
                for (int k = 0; k < cur->sons.size(); ++k) {
                    if(cur->sons[k]->description == path[j]){
                        index = k;
                        break;
                    }
                }
                if(index == cur->sons.size()){
                    cur->sons.push_back(new trieTree(path[j]));
                    result++;
                    cur = cur->sons[cur->sons.size()-1];
                }else{
                    cur = cur->sons[index];
                }
            }
        }
    }
    return root;
}

trieTree* reduceTree(trieTree* root){
    trieTree* re = root;
    if(root->sons.empty()) return root;
    if(root->sons.size() >= 2){
        for (int i = 0; i < root->sons.size(); ++i) {
            reduceTree(root->sons[i]);
        }
    }
    if(root->sons.size() == 1){
        trieTree* next = root->sons[0];
        if(next->sons.empty()){
//            root->description = root->description + "-";
//            root->description = root->description + next->description;
//            root->sons.clear();
            return root;
        }
        root->sons.clear();
        for (int i = 0; i < next->sons.size(); ++i) {
            root->sons.push_back(next->sons[i]);
        }
        root->description = root->description + "-";
        root->description = root->description + next->description;
        reduceTree(root);
    }
    return re;
}

void printTree(trieTree* root,string out){
    if(nullptr == root) return;
    out += root->description + "/";
    if(root->sons.empty()){
        if(out.find_last_of('/') == out.size() - 1){
            out.erase(out.size()-1);
        }
        cout<<out<<endl;
    }else{
        for (int i = 0; i < root->sons.size(); ++i) {
            printTree(root->sons[i],out);
        }
    }
}
void solve(vector<string> &pathes, int n){
    int result = 0;
    trieTree* root = createTree(pathes);
    trieTree* reduced = reduceTree(root);
    string out = "";
    printTree(reduced,out);
}

int main(){
    freopen("d:/A.in","r",stdin);
    int n;
    cin>>n;
    vector<string> pathes(n,"");
    for (int i = 0; i < n; ++i) {
        cin >> pathes[i];
    }
    solve(pathes,n);
    return 0;
}

```
