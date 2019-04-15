---
title: 算法专题_二叉树2
toc: true

tags:
  - ACM
date: 2017-09-23 18:26:58
---

【前序遍历】【中序遍历】【后序遍历】【路径搜索】【公共祖先】【堆】【top K】【线段树】【树状数组】【并查集】

<!-- more -->

### [前序遍历](http://lintcode.com/zh-cn/problem/binary-tree-preorder-traversal/)

```c
class TreeNode {
public:
    int val;
    TreeNode *left, *right;
    TreeNode(int val) {
        this->val = val;
        this->left = this->right = NULL;
    }
};
// 递归前序遍历
vector<int> preorderTraversal(TreeNode * root) {
    vector<int> re;
    if(!root) return re;

    re.push_back(root->val);
    if(root->left){
        auto tmp = preorderTraversal(root->left);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    if(root->right){
        auto tmp = preorderTraversal(root->right);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    return re;
}

// 非递归前序遍历
vector<int> preorder(TreeNode* root){
    vector<int> re;
    if(!root) return re;

    stack<TreeNode*> s;
    TreeNode* iter = root;
    while(iter || !s.empty()){
        while(iter){
            // 首先输出根
            re.push_back(iter->val);
            s.push(iter);
            iter = iter->left;
        }
        iter = s.top();
        s.pop();
        iter = iter->right;
    }
    return re;
}
```

### [中序遍历](http://lintcode.com/zh-cn/problem/binary-tree-inorder-traversal/)

```c
// 递归中序遍历
vector<int> inorderTraversal(TreeNode *root) {
    vector<int> re;
    if(!root) return re;

    if(root->left){
        auto tmp = inorderTraversal(root->left);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    re.push_back(root->val);
    if(root->right){
        auto tmp = inorderTraversal(root->right);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    return re;
}

// 非递归中序遍历
vector<int> inorder(TreeNode* root){
    vector<int> re;
    if(!root) return re;

    stack<TreeNode*> s;
    TreeNode* iter = root;
    while(iter || !s.empty()){
        while(iter){
            s.push(iter);
            iter = iter->left;
        }
        iter = s.top();
        s.pop();
        // 输出
        re.push_back(iter->val);
        iter = iter->right;
    }
    return re;
}
```

### [后序遍历](http://lintcode.com/zh-cn/problem/binary-tree-postorder-traversal/)

```c
// 递归后序遍历
vector<int> postorderTraversal(TreeNode *root) {
    vector<int> re;
    if(!root) return re;

    if(root->left){
        auto tmp = postorderTraversal(root->left);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    if(root->right){
        auto tmp = postorderTraversal(root->right);
        re.insert(re.end(),tmp.begin(),tmp.end());
    }
    re.push_back(root->val);
    return re;
}
// 后序遍历的非递归实现
vector<int> postOrder(TreeNode *root)
{
    vector<int> re;
    if(!root) return re;

    TreeNode *iter = root;
    stack<TreeNode *> s;
    TreeNode *last = root;
    s.push(iter);
    while (!s.empty())
    {
        iter = s.top();
        // 1. 当前节点是叶节点(没有左右孩子)，输出
        // 2. 上次访问了当前节点的左孩子，而且当前节点没有右孩子，输出
        // 3. 上次访问了当前节点的右孩子，输出
        if( (iter->left == nullptr && iter->right == nullptr) ||
                (iter->right == nullptr && last == iter->left) ||
                    (last == iter->right) )
        {
            re.push_back(iter->val);
            last = iter;
            s.pop();
        }
        else
        {
            // 栈是后进先出的，要先处理左孩子，所以要先放入右耗资
            if(iter->right)
                s.push(iter->right);
            if(iter->left)
                s.push(iter->left);
        }
    }
    return re;
}
```

### 二叉树的路径搜索

```c
// 搜索根节点到叶子节点的所有路径 [http://lintcode.com/zh-cn/problem/binary-tree-paths/#]
vector<vector<int>> vv;
void binaryTreePaths(TreeNode *root,vector<int> &tmp) {
    // write your code here
    if(!root->left && !root->right){
        tmp.push_back(root->val);
        vv.push_back(tmp);
        tmp.clear();
        return;
    }else{
        tmp.push_back(root->val);
        vector<int> tmp1(tmp);
        vector<int> tmp2(tmp);
        if(root->left) binaryTreePaths(root->left,tmp1);
        if(root->right) binaryTreePaths(root->right,tmp2);
    }
}
```

### [二叉树最大深度](http://lintcode.com/zh-cn/problem/maximum-depth-of-binary-tree/)

```c
// 调用方式： maxDepth2(root,0)
int maxDepth2(TreeNode *root, const int dep){
    if(!root) return dep;
    if(!root->left && !root->right) return dep+1;
    if(root->left && !root->right) return maxDepth2(root->left,dep+1);
    if(root->right && !root->left) return maxDepth2(root->right,dep+1);     
    return max(maxDepth2(root->left,dep+1),maxDepth2(root->right,dep+1));   
}
```

### [二叉树最小深度](http://lintcode.com/zh-cn/problem/minimum-depth-of-binary-tree/#)

```c
// 最小深度 
int minDepth2(TreeNode* root,const int dep){
    if(!root) return dep;
    if(!root->left && !root->right) return dep+1;
    if(root->left && !root->right)  return minDepth2(root->left,dep+1);
    if(!root->left && root->right)  return minDepth2(root->right,dep+1);
    return min(minDepth2(root->left,dep+1),minDepth2(root->right,dep+1));
}
```

###  叶子节点到叶子节点路径的搜索

```c
map<TreeNode*,vector<TreeNode*>> nodes; // 存储树的结构
map<TreeNode*,bool> visited;
vector<vector<int>> vv;
void dfs(TreeNode* leaf,vector<int> &v){
    visited[leaf] = true;
    vector<TreeNode*> tmp = nodes[leaf];
    bool is = true;
    v.push_back(leaf->val);
    for(int i=0;i<tmp.size();i++){
        if(visited[tmp[i]] == false){
            vector<int> v1(v);
            dfs(tmp[i],v1);
            is = false;
        }
    }
    if(is){ // 叶节点
        vv.push_back(v);
    }
    return;
}
```

### top K 问题

如果是最大的K个数，那么首先使用前K个数建立一个最小堆，然后从第K+1个数开始和堆顶的元素比较，如果大于堆顶的元素，就把堆顶的元素弹出，把该元素push进堆。

```c
vector<int> topk(vector<int> &v, int k){
    priority_queue<int,vector<int>,greater<int>> heap; // 最小堆（堆顶元素最小）
    for(int i=0;i<k;i++){
        heap.push(v[i]);
    }
    for(int i=k;i<v.size();i++){
        if(v[i] > heap.top()){
            heap.pop();
            heap.push(v[i]);
        }

    }
    vector<int> re;
    while(!heap.empty()){
        re.push_back(heap.top());
        heap.pop();
    }
    return re;
}
```

### [数据流的中位数](http://lintcode.com/zh-cn/problem/data-stream-median/)

使用大顶堆(`priority_queue<int,vector<int>,less<int>>`) 保存前半部分的数据，使用小顶堆(`priority_queue<int,vector<int>,greater<int>>`) 保存后半部分的数据, 保证大顶堆的顶 小于等于 小顶堆的顶， 那么在每次需要求当前的数据的中位数的时候，只需要两个堆顶元素就可以求出来了。

```c
class MedianFinder {
private:
    priority_queue<int,vector<int>,less<int>> maxHeap;
    priority_queue<int,vector<int>,greater<int>> minHeap;
    long long n;
    bool is_left;
public:
    MedianFinder() {
        this->n = 0;
        this->is_left = true;
    }

    void addNum(int num) {
        // 轮流加入左右两个堆
        if(is_left){
            // 如果right的堆为空，可以确保左边最大的小于等于右边最小的，直接加入
            // 如果要加入的元素小于等于right的堆顶，也可以保证上述条件
            if(minHeap.empty() || num <= minHeap.top()){
                maxHeap.push(num);
            }
            // 如果num > minHeap.top(); 不能直接加入maxHeap,因为如果这样做，就不能保证maxHeap中的所有的数字都小于等于minHeap中的数字。
            // 这个时候的做法是， 先把num加入minHeap, 在把minHeap.top()弹出，加入maxHeap;
            else{
                minHeap.push(num);
                maxHeap.push(minHeap.top());
                minHeap.pop();
            }
            is_left = !is_left;
        }
        if(!is_left){
            if(maxHeap.empty() || num >= maxHeap.top()){
                minHeap.push(num);
            }else{
                maxHeap.push(num);
                minHeap.push(maxHeap.top());
                maxHeap.pop();
            }
            is_left = !is_left;
        }
        this->n++;
    }

    double findMedian() {
        if(n >> 1){
            return (maxHeap.top() + minHeap.top()) / 2.0;
        }else{
            return maxHeap.top();
        }
    }
};
```

### [合并K个排序链表](http://lintcode.com/zh-cn/problem/merge-k-sorted-lists)

这里使用小顶堆这种结构保存K个链表，每次取得堆顶的元素就可以得到K个链表中最小的。

```c
ListNode *mergeKLists(vector<ListNode *> &lists) {
    priority_queue<ListNode*,vector<ListNode*>,bigger> minHeap;
    if(lists.empty()) return nullptr;
    for(int i=0;i<lists.size();i++){
        if(lists[i]) minHeap.push(lists[i]);
    }
    ListNode *head = nullptr;
    ListNode *iter = nullptr;
    while(!minHeap.empty()){
        if(head == nullptr){
            head = minHeap.top();
            iter = head;
            if(minHeap.top()->next){
                minHeap.push(minHeap.top()->next);
            }
            minHeap.pop();
        }else{
            iter->next = minHeap.top();
            iter = iter->next;
            if(minHeap.top()->next){
                minHeap.push(minHeap.top()->next);
            }
            minHeap.pop();
        }
    }
    return head;
}
```

### 堆排序

使用堆这种结构维护数据的有序性，从而进行排序的算法是堆排序。堆排序的具体步骤是： 使用要排序的整个数组建立堆。然后把最后一个元素和堆顶元素交换，堆的大小减1，调整堆，然后再把新的堆的最后一个元素和堆顶元素交换，调整堆，直到全部数据有序为止。

```c
void heapSort(vector<int> &v){
    // 建立最大堆
    make_heap(v.begin(),v.end(),less<int>());
    int index = v.size();
    while(index > 0){
        // 交换堆顶和堆尾的元素，重新调整堆
        pop_heap(v.begin(),v.begin()+index,less<int>());
        index--;
    }
}
```

### 二叉树的序列化和反序列化

```c

#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <queue>

using namespace std;

template <typename T>
struct Node{
    T val;
    Node  *left,*right;
    Node(){}
    Node(T init) : val(init), left(nullptr), right(nullptr) {}
};
/**
 * 二叉树的先根遍历，非递归实现，使用栈
 * @param root
 * @return
 */
vector<int> preOrder(Node<int>* root){
    vector<int> ans;
    if(!root) return ans;
    stack<Node<int>*> s;
    Node<int>* iter = root;
    while(iter || !s.empty()){
        while(iter){
            ans.push_back(iter->val);
            s.push(iter);
            iter = iter->left;
        }
        iter = s.top();
        s.pop();
        iter = iter->right;
    }
    return ans;
}
/**
 * 二叉树的层次遍历，非递归，使用队列
 * @param root
 * @return
 */
vector<int> levelOrder(Node<int>* root){
    vector<int> ans;
    if(!root) return ans;
    queue<Node<int>*> q;
    q.push(root);
    Node<int>* iter;
    while(!q.empty()){
        iter = q.front();
        if(iter){
            ans.push_back(iter->val);
            if(iter->left) q.push(iter->left);
            if(iter->right) q.push(iter->right);
        }
        q.pop();
    }
    return ans;
}

/**
 * 使用层次遍历的方法序列化二叉树
 * @param root
 * @return
 */
vector<string> levelOrderSeries(Node<int>* root){
    vector<string> ans;
    if(!root) return ans;
    queue<Node<int>*> q;
    q.push(root);
    Node<int>* iter;
    while(!q.empty()){
        iter = q.front();
        if(iter){
            if(iter->val != -1){
                ans.push_back(to_string(iter->val));
                if(iter->left){
                    q.push(iter->left);
                }else{
                    q.push(new Node<int>(-1));
                }
                if(iter->right){
                    q.push(iter->right);
                }else{
                    q.push(new Node<int>(-1));
                }
            }else{
                ans.push_back("#");
            }
        }
        q.pop();
    }
    while(*(--ans.end()) == "#") ans.erase(--ans.end());
    return ans;
}
/**
 * 根据二叉树的层次遍历的结果重建二叉树
 * @param c
 * @return
 */
Node<int>* reConstructTree(const vector<string> &c){
    if(c.empty()) return nullptr;

    queue<Node<int>*> q;
    int index = 0;
    if(c[index] == "#") return nullptr;
    auto root = new Node<int>( stoi(c[index]) );  index++;
    q.push(root);

    while(!q.empty()){
        Node<int>* parent = q.front(); q.pop();
        Node<int>* left = nullptr;
        Node<int>* right = nullptr ;
        if(index < c.size() && c[index] != "#") { left = new Node<int>(stoi(c[index]));  }  index++;
        if(index < c.size() && c[index] != "#") { right = new Node<int>(stoi(c[index])); }  index++;
        if(left) { parent->left = left; q.push(left); }
        if(right) { parent->right = right; q.push(right); }
    }
    return root;
}
int main(){
    vector<string> v = {"3","9","20","#","#","15","7"};
    auto re = reConstructTree(v);

    auto hh = levelOrderSeries(re);
    return 0;

}
```

### 使用非递归、非栈的方式遍历二叉树

核心的思想是使用二叉树的左子树的最右边的空指针指向  在遍历完左子树之后 需要遍历的下一个节点。 具体的步骤是：

1. 令`iter = root`,`root`表示二叉树的根节点。首先让`iter`的左子树的最右节点的`right`指针指向`iter`. 完成这一步的代码类似下面这样：

```c
Node* tp = iter->left; // tp 指向iter的左子树
if(tp){
    // 通过循环找到最右边的节点
    while(tp->right != nullptr){
         tp = tp->right;
    }
    // 最右节点的right指针指向iter
    if(tp->right == nullptr){
        tp->right = iter; 
}
```

2. 处理完`root`节点之后，循环处理`iter=iter->left`直到`iter`的`left`指针为空为止，也就是上文中的`tp==nullptr`的情况。


```c
#include <iostream>
#include <vector>
#include <map>

using namespace std;

struct Node{
    int val;
    Node* left;
    Node* right;
    Node(){}
    Node(int v) : val(v),left(nullptr),right(nullptr){}
};

/**
 * 二叉树的前序遍历，非递归，不用栈
 * @param root
 * @return
 */
vector<int> preOrder(Node* root){
    vector<int> ans;
    if(!root) return ans;

    Node* iter = root;
    Node* tp = nullptr;
    while(iter != nullptr){
        // 处理iter的左子树
        tp = iter->left;
        if(tp != nullptr){
            // 找到 iter 左子树的最右节点
            while(tp->right != nullptr && tp->right != iter){
                tp = tp->right;
            }
            // 如果是不满足tp->right != nullptr 而退出了上面的循环，
            // 说明找到的节点iter的左子树的最右面的空节点，
            // 把该节点指向iter
            // 继续处理 iter 的左孩子
            if(tp->right == nullptr){
                tp->right = iter; // 左子树的最右节点指向之前的节点
                ans.push_back(iter->val); // 一旦找到左子树最右边的节点就输出iter，因为前序遍历根节点要最先输出
                iter = iter->left;
                continue;
            }
            // 如果是因为不满足tp->right != iter 退出了循环，
            // 就是说tp->right == iter
            // 说明已经遍历过一遍(否则不可能有指向iter的指针)
            // 把tp->right 置空，恢复二叉树原来的样子
            else{
                tp->right = nullptr;
            }
        }else{
            // 没有左子树，也要输出根
            ans.push_back(iter->val);
        }
        iter = iter->right;
    }
    return ans;
}

/**
 * 二叉树的中序遍历，非递归，不用栈
 * @param root
 * @return
 */
vector<int> inOrder(Node* root){
    vector<int> ans;
    if(!root) return ans;

    Node* iter = root;
    Node* tp = nullptr;
    while(iter != nullptr){
        // 处理iter的左子树
        tp = iter->left;
        if(tp != nullptr){
            // 找到 iter 左子树的最右节点
            while(tp->right != nullptr && tp->right != iter){
                tp = tp->right;
            }
            // 如果是不满足tp->right != nullptr 而退出了上面的循环，
            // 说明找到的节点iter的左子树的最右面的空节点，
            // 把该节点指向iter
            // 继续处理 iter 的左孩子
            if(tp->right == nullptr){
                tp->right = iter; // 左子树的最右节点指向之前的节点
                iter = iter->left;
                continue;
            }
            // 如果是因为不满足tp->right != iter 退出了循环，
            // 就是说tp->right == iter
            // 说明已经遍历过一遍(否则不可能有指向iter的指针)
            // 把tp->right 置空，恢复二叉树原来的样子
            else{
                tp->right = nullptr;
            }
        }
        ans.push_back(iter->val);
        iter = iter->right;
    }
    return ans;
}

int main(){

    return 0;
}

```