---
title: 算法专题_栈和队列
toc: true

tags:
  - ACM
date: 2017-04-12 10:07:18
---

栈是一种实现了后进先出的规则的数据结构。C++里面有实现`stack`.使用的时候`include <stack>`,支持的操作有`push(),pop(),top()`;需要注意的是，`pop`的作用是弹出栈顶元素，而不会返回栈顶元素，实现返回栈顶元素并弹出的操作应该是`int a = s.top(); s.pop();`.

队列是一种先进先出的数据结构。C++里面有两种实现， 一种是普通的队列`queue`,使用的时候`include<queue>`,支持的操作和栈类似，`push(),pop(),top()`分别实现数据的入队，出队和读取队列的头部元素。`deque`实现了双端队列，支持从两端入队和出队.使用的时候`include<deque>`。

<!-- more -->

### 有效的括号序列

[有效的括号序列](http://lintcode.com/zh-cn/problem/valid-parentheses/)

使用栈存储所有的左括号，一旦遇到右括号就与栈顶的元素比较，如果匹配，弹出栈顶元素，继续，如果不匹配，直接返回false.

```c
bool isValidParentheses(string& s) {
    // 括号匹配问题
    stack<char> cc;
    map<char,char> m;
    m['('] = ')';
    m['['] = ']';
    m['{'] = '}';
    for(int i=0;i<s.size();++i){
        if(s[i] == '(' || s[i] == '['|| s[i]=='{'){
            cc.push(s[i]);
        }
        if(s[i] == ')' || s[i] == ']' || s[i] == '}'){
            char tmp = cc.top();
            if(m[tmp] != s[i]){
                return false;
            }
            cc.pop();
        }
    }
    if(cc.empty()) return true;
    return false;
}
```

### 逆波兰表达式求值

求逆波兰表达式的值。在逆波兰表达法中，其有效的运算符号包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰计数表达。

[逆波兰表达式求值](http://lintcode.com/zh-cn/problem/evaluate-reverse-polish-notation/)

```c
// 运算符只考虑了加减乘除
template<typename T>
T compute(T first, T second,string op){
    if(op == "+") return first + second;
    if(op == "-") return first - second;
    if(op == "*") return first * second;
    if(op == "/" && second != 0) return first / second;
    return -1000; // 说明输入的数据有问题
}
int evalRPN(vector<string> &tokens) {
    // write your code here
    stack<int > s;
    set<string> op = {"+","-","*","/"}; // 都是双目运算符，处理逻辑是相同的
    for(size_t i=0;i<tokens.size();i++){
        if(op.count(tokens[i]) > 0){
            int first=0,second=0;
            if(!s.empty()){ first = s.top() ; s.pop(); }
            if(!s.empty()){ second = s.top() ; s.pop(); }
            s.push(compute(second,first,tokens[i]));
        }else{
            s.push(stoi(tokens[i]));
        }
    }
    return s.top();
}
```

### 中缀表达式转换成后缀表达式

利用栈将中缀表达式转换成后缀表达式的步骤(逆波兰表达式)：

1. 初始化栈cc为空用来存放运算符，初始化string s为空用来存储转换之后的逆波兰表达式；
2. 从头到尾扫描中缀表达式，遇到数字直接输出到s.
3. 遇到运算符，如果当前的栈cc为空，则直接入栈；如果遇到的是左括号，直接入栈；如果遇到的运算符的优先级低于栈顶的元素，直接入栈；
4. 如果遇到的运算符的优先级小于等于当前栈顶的元素，那么持续输出栈顶元素，直到栈顶元素的运算优先级小于当前元素 或者 栈为空 为止。循环退出后将当前运算符入栈。
5. 如果遇到的是右括号，持续输出栈顶元素，直到遇到左括号，左括号弹出，但不输出到s.

[将表达式转换为逆波兰表达式](http://lintcode.com/zh-cn/problem/convert-expression-to-reverse-polish-notation/#)

```c
bool isNumber(string &s){
    // 1. 空字符串不是数字
    if(s.empty()) return false;
    if(s.size() == 1 && s[0] == '-') return false;
    for(size_t i=0;i<s.length();i++){
        // 2. 如果第二个字符及以后的不是0--9，不是数字，直接返回false
        if( (i>0 && (s[i]<'0'||s[i]>'9')) ) return false;
        // 3. 如果第一个字符不是0--9，也不是-，则不是数字，直接返回false.
        if(i==0&&(s[i]!='-')&&(s[i]<'0'||s[i]>'9') ) return false;
    }
    return true;
}
vector<string> convertToRPN(vector<string> &expression) {
    // write your code here
    vector<string> ans;
    if(expression.empty()) return ans;

    // 操作符
    stack<string> s;
    // 操作符的优先级
    map<string,int> kv; 
    kv["+"] = 3; kv["-"] = 3; 
    kv["*"] = 4; kv["/"] = 4; 
    kv["("] = 2; kv[")"] = 6;
    for(size_t i=0;i<expression.size();i++){
        // 1. 如果是数字，直接输出
        if(isNumber(expression[i])){
            ans.push_back(expression[i]);
        }
        // 2. 如果是符号+,-,*,/,(,),需要按照优先级顺序处理
        else if(kv[expression[i]] >= 2 && kv[expression[i]] <= 6){
            // 2.1 如果栈为空，或者是左括号，直接入栈
            if(s.empty() || kv[expression[i]]==2 ){
                s.push(expression[i]);
            }
            // 2.2 如果遇到的是右括号，持续输出栈中的元素，直到遇到左括号为止
            else if(kv[expression[i]] == 6){
                while(!s.empty() && s.top() != "("){
                    ans.push_back(s.top()); s.pop();
                }
                if(!s.empty()) s.pop(); // 弹出左括号
            }
            // 如果是+,-,*,/，就要比较和当前栈顶元素的优先级了，如果比栈顶运算符的优先级低，可以直接入栈，
            // 如果高，需要弹出栈顶元素，直到栈顶的运算符低于当前的运算符或者栈为空为止
            else if(kv[expression[i]]==3 || kv[expression[i]]==4){
                 // 2.3 如果当前的运算符优先级高于栈顶的，直接入栈
                 if(kv[expression[i]] > kv[s.top()]){
                     s.push(expression[i]);
                 }else{
                     while(!s.empty() && kv[expression[i]] <= kv[s.top()]){
                         ans.push_back(s.top()); s.pop();
                     }
                     s.push(expression[i]);
                 }
            }
        }
    }
    // 把栈中的操作符输出
    while(!s.empty()){
        ans.push_back(s.top()); s.pop(); 
    } 
    return ans;
}
```

### 中缀表达式求值

先转换成后缀表达式，再求值。

[表达式求值](http://lintcode.com/zh-cn/problem/expression-evaluation/)

```c
int evaluateExpression(vector<string> expression) {
    // write your code here
    vector<string> r = convert(expression);
    if(r.empty()) return  0;
    return evalRPN(r);
}
```

### 表达树构造

表达树是一个二叉树的结构，用于衡量特定的表达。所有表达树的叶子都有一个数字字符串值。而所有表达树的非叶子都有另一个操作字符串值。

给定一个表达数组，请构造该表达的表达树，并返回该表达树的根。

[表达树构造](http://lintcode.com/zh-cn/problem/expression-tree-build/#)

思路： 可以利用以前做过的题目，将本题分解成几个子问题。通过分析我们发现，中缀表达式是表达树的中序遍历，后缀表达式是表达树的后序遍历，通过中缀表达式我们可以得到后缀表达式，这个在之前已经做过。而通过二叉树的中序遍历序列和后序遍历序列也可以很容易的重建出该二叉树。所以该问题分解为这样几个子问题：

1. 求后缀表达式
2. 根据中序遍历和后序遍历序列重建二叉树

需要注意的是，本题中以数字和运算符作为节点的值，这并不能保证节点值的唯一，而重建二叉树的时候要求节点值唯一。这里我们采用定义新结构的方法使得每个节点都是唯一的。具体见代码。

```c
struct uniqueNode{
    string description;
    int id;
    uniqueNode():description(""), id(0){}
    uniqueNode(string s, int i) : description(s), id(i){}
};

//Definition of ExpressionTreeNode:
class ExpressionTreeNode {
public:
    string symbol;
    ExpressionTreeNode *left, *right;
    ExpressionTreeNode(string symbol) {
        this->symbol = symbol;
        this->left = this->right = NULL;
    }
};

//后序遍历和中序遍历重建二叉树
ExpressionTreeNode* buildTree(vector<uniqueNode>& inorder, vector<uniqueNode>& postorder) {
    if (inorder.empty() || postorder.empty()) { return nullptr; }
    ExpressionTreeNode *node = new ExpressionTreeNode(postorder[postorder.size()-1].description);// 先建立根节点
    int len = inorder.size();
    for (size_t i = 0; i < inorder.size(); ++i) {
        if (inorder[i].id == postorder[postorder.size()-1].id) {
            // 找到了中序遍历的根节点，则左面的全部是左子树，右面的全部是右子树。
            vector<uniqueNode> left_post(i);
            vector<uniqueNode> left_vin(i);
            vector<uniqueNode> right_post(len - 1 - i);
            vector<uniqueNode> right_vin(len - 1 - i);
            copy(inorder.begin(), inorder.begin() + i, left_vin.begin());
            copy(inorder.begin() + i + 1, inorder.end(), right_vin.begin());
            copy(postorder.begin(), postorder.begin() + i, left_post.begin());
            copy(postorder.begin() + i, postorder.end()-1, right_post.begin());
            node->left = buildTree(left_vin,left_post);
            node->right = buildTree(right_vin,right_post);
        }
    }
    return node;
}

inline bool isnumber(string &s){
    for(auto a : s){
        if(a > '9' || a < '0'){
            return false;
        }
    }
    return true;
}
/**
 * 中缀表达式 -> 后缀表达式
 * @param expression
 * @return
 */
vector<uniqueNode> convert(vector<uniqueNode> expression) {
    vector<uniqueNode> re;
    stack<uniqueNode> cc;
    // 设置运算符的优先级
    map<string,int> p;
    p["+"] = 3;
    p["-"] = 3;
    p["*"] = 4;
    p["/"] = 4;
    p["("] = 2;
    p[")"] = 1;
    for(int i=0; i< expression.size();i++){
        // 遇到数字直接入栈
        if(isnumber(expression[i].description)){
            re.push_back(expression[i]);
            continue;
        }
        // 遇到操作符
        if(p[expression[i].description] >= 1 && p[expression[i].description] <= 4){
            //栈为空或者遇到左括号，直接入栈
            if(cc.empty() || p[expression[i].description] == 2 || p[expression[i].description] > p[cc.top().description]){
                cc.push(expression[i]);
            }
                //遇到右括号，弹出栈中左括号之前的所有运算符号，输出，最后左括号弹出，不输出
            else if(p[expression[i].description] == 1){
                while(cc.top().description != "("){
                    auto tmp = cc.top();
                    re.push_back(tmp);
                    cc.pop();
                }
                cc.pop(); // 弹出左括号
            }
                //当前运算符的优先级小于等于栈顶运算符的优先级，应该先计算栈顶的运算符，所以弹出栈顶的运算符，输出，把当前的运算符入栈
            else if(p[expression[i].description] <= p[cc.top().description]){
                // 弹出栈中优先级高于当前运算符的运算符
                while(!cc.empty() && p[expression[i].description] <= p[cc.top().description]){
                    re.push_back(cc.top());
                    cc.pop();
                }
                // 当栈为空 或者 遇到运算符的运算顺序低于当前运算符的时候，循环退出
                cc.push(expression[i]);
                //当前运算符的优先级大于栈顶运算符的优先级，直接入栈
            }
            continue;
        }
        cout<<"error 1."<<endl;
    }
    while(!cc.empty()){
        re.push_back(cc.top());
        cc.pop();
    }
    return re;
}

ExpressionTreeNode* build(vector<string> &expression) {
    vector<uniqueNode> v;
    for (int i = 0; i < expression.size(); ++i) {
        v.push_back(uniqueNode(expression[i],i+1));
    }
    auto last = convert(v);
    for (int j = 0; j < v.size(); ++j) {
        if(v[j].description == "(" || v[j].description == ")"){
            v.erase(v.begin()+j);
            j--;
        }
    }
    auto head = buildTree(v,last);
    return head;
}
```

表达树的特殊结构决定了我们只根据后缀表达式也能构造出来一棵表达树(注意：只根据后序遍历序列是不能构造一棵二叉树的，因为答案不唯一。)，具体的做法是模拟后缀表达式的计算顺序。

```c
//Definition of ExpressionTreeNode:
class ExpressionTreeNode {
public:
    string symbol;
    ExpressionTreeNode *left, *right;
    ExpressionTreeNode(string symbol) {
        this->symbol = symbol;
        this->left = this->right = NULL;
    }
};
bool isnumber(string &s){
    for (int i = 0; i < s.size(); ++i) {
        if(s[i] < '0' || s[i] > '9') return false;
    }
    return true;
}
bool isoperator(string &s){
    if(s == "+" || s == "-" || s == "*" || s == "/") return true;
    return false;
}
vector<string> convert(vector<string> expression) {
    vector<string> re;
    stack<string> cc;
    // 设置运算符的优先级
    map<string,int> p;
    p["+"] = 3;
    p["-"] = 3;
    p["*"] = 4;
    p["/"] = 4;
    p["("] = 2;
    p[")"] = 1;
    for(int i=0; i< expression.size();i++){
        // 遇到数字直接入栈
        if(isnumber(expression[i])){
            re.push_back(expression[i]);
            continue;
        }
        // 遇到操作符
        if(p[expression[i]] >= 1 && p[expression[i]] <= 4){
            //栈为空或者遇到左括号，直接入栈
            if(cc.empty() || p[expression[i]] == 2 || p[expression[i]] > p[cc.top()]){
                cc.push(expression[i]);
            }
                //遇到右括号，弹出栈中左括号之前的所有运算符号，输出，最后左括号弹出，不输出
            else if(p[expression[i]] == 1){
                while(cc.top() != "("){
                    string tmp = cc.top();
                    re.push_back(tmp);
                    cc.pop();
                }
                cc.pop(); // 弹出左括号
            }
                //当前运算符的优先级小于等于栈顶运算符的优先级，应该先计算栈顶的运算符，所以弹出栈顶的运算符，输出，把当前的运算符入栈
            else if(p[expression[i]] <= p[cc.top()]){
                // 弹出栈中优先级高于当前运算符的运算符
                while(!cc.empty() && p[expression[i]] <= p[cc.top()]){
                    re.push_back(cc.top());
                    cc.pop();
                }
                // 当栈为空 或者 遇到运算符的运算顺序低于当前运算符的时候，循环退出
                cc.push(expression[i]);
                //当前运算符的优先级大于栈顶运算符的优先级，直接入栈
            }
            continue;
        }
        cout<<"error 1."<<endl;
    }
    while(!cc.empty()){
        re.push_back(cc.top());
        cc.pop();
    }
    return re;
}
ExpressionTreeNode* build(vector<string> &expression) {
    expression = convert(expression);
    if(expression.empty())
        return nullptr;
    stack<ExpressionTreeNode*> cc;
    for (int i = 0; i < expression.size(); ++i) {
        if(isnumber(expression[i])){
            ExpressionTreeNode *node = new ExpressionTreeNode(expression[i]);
            cc.push(node);
        }
        if(isoperator(expression[i])){
            ExpressionTreeNode *node = new ExpressionTreeNode(expression[i]);
            ExpressionTreeNode *right = cc.top();
            cc.pop();
            ExpressionTreeNode *left = cc.top();
            cc.pop();
            node->left = left;
            node->right = right;
            cc.push(node);
        }
    }
    return cc.top();
}
```

### 简易的正则表达式解析

> - s = abc3[a], return abcaaa
> - s = 3[abc], return abcabcabc
> - s = 4[ac]dy, return acacacacdy
> - s = 3[2[ad]3[pf]]xyz, return adadpfpfpfadadpfpfpfadadpfpfpfxyz

[Expression Expand](http://lintcode.com/zh-cn/problem/expression-expand/)

```c
string expressionExpand(string& s) {
    bool finished = true;
    for(int i = 0; i < s.length(); ++i){
        if(s[i]  == '[' || s[i] == ']'){
            finished = false;
            break;
        }
    }
    if(finished) return s;
    int left, right,pre_left;
    for(int i=0;i<s.length();++i){
        if(s[i] == '['){
            left = i;
        }
        if(s[i] == ']'){
            for(int j = left-1; j >= 0;--j){
                if(s[j]>='0' && s[j]<='9'){
                    pre_left = j;
                }else{
                    break;
                }
            }

            string num = s.substr(pre_left,left-pre_left);
            int count = stoi(num);
            string cycle = s.substr(left+1,i-left-1);
            string cycle_with_count = "";
            while(count--){
                cycle_with_count += cycle;
            }
            s.erase(pre_left,i-pre_left+1);
            s.insert(pre_left,cycle_with_count);
            return expressionExpand(s);
        }
    }
    return expressionExpand(s);
}
```

### 带有getMin()的栈


实现一个可以在O(1)时间内得到最小值的栈结构。支持`pop(),push(),top(),getMin()`四种操作。

[带最小值操作的栈](http://lintcode.com/zh-cn/problem/min-stack/)

**思路**  使用两个栈来模拟这个栈，一个栈用来保存实际的数据，一个栈用来保存对应数据的最小值。每次入栈和出栈操作的时候，判断最小值有没有发生变化，如果没有发生变化，直接弹出数据，如果发生了变化，更新存储最大值最小值的栈。

```c
class StackWithGetMin2{
public:
    stack<int> data;
    stack<int> min_data;
    void push(int val){
        data.push(val);
        if(!min_data.empty()){
            if(min_data.top()>=val){
                min_data.push(val);
            }
        }
        else{
           min_data.push(val);
       }
    }
    int pop(){
        auto v = data.top();
        if(v > min_data.top()){
            data.pop();
            return v;
        }else{
            data.pop();
            min_data.pop();
            return v;
        }
    }
    int getMin(){
      if(min_data.empty){throw new exception;}
      else{
         return min_data.top();
      }
    }
    int top(){
        return data.top();
    }
};
```


### 132序列

Given a sequence of n integers a1, a2, ..., an, a 132 pattern is a subsequence ai, aj, ak such that i < j < k and ai < ak < aj. Design an algorithm that takes a list of n numbers as input and checks whether there is a 132 pattern in the list.

n will be less than 20,000.

[pattern](http://lintcode.com/zh-cn/problem/132-pattern/#)

```c
bool find132pattern(vector<int>& nums) {
    int third = INT32_MIN;
    stack<int> s;
    for (int i = nums.size() - 1; i >= 0; --i) {
        if (nums[i] < third) return true;
        else while (!s.empty() && nums[i] > s.top()) {
                third = s.top(); s.pop();
            }
        s.push(nums[i]);
    }
    return false;
}
```

### 用栈实现队列

[用栈实现队列]()

```c
class MyQueue {
public:
    stack<int> stack1;
    stack<int> stack2;

    MyQueue() {
        // do intialization if necessary
    }

    void push(int element) {
        // write your code here
        stack1.push(element);
    }

    int pop() {
        // write your code here
        if(stack2.empty()){
            if(stack1.empty()){
               return -1;
            }
            while(!stack1.empty()){
                int a = stack1.top();
                stack1.pop();
                stack2.push(a);
            }
        }
        int a = stack2.top();
        stack2.pop();
        return a;
    }

    int top() {
        // write your code here
        if(stack2.empty()){
            if(stack1.empty()){
               return -1;
            }
            while(!stack1.empty()){
                int a = stack1.top();
                stack1.pop();
                stack2.push(a);
            }
        }
        int a = stack2.top();
        return a;
    }
};

```

### 用一个栈排序另外一个栈

要求排序好的数据就在原来的栈里面。

**思路**  设存放数据的栈是`s`, 辅助排序的栈是`help`,一个临时变量`v`用来暂存从栈里面出来的数据。我们只要确保把`s`中的数有序的入栈`help`,然后在反向从`help`压入`s`就能得到需要的结果。如果要求从大到小排序，即`s`栈顶的元素是最大的，那么`help`中就应该是从小到大排序,栈顶的元素是最小的。每次从`s`向`help`移动数据的时候，都要和`help`的栈顶比较，如果小于栈顶的元素，直接入栈，如果大于栈顶的元素，把栈顶元素入栈到`s`,直到`v`中的数小于`help`的栈顶元素，才将`v`入栈到`help`,其实这有点向汉诺塔问题，循环利用两个栈做中转，只不过更加简单。

```c
void sortStack(stack<int> s){
    stack<int> help;
    help.push( s.top() );
    s.pop();
    auto v = s.top();
    if(help.top()>v){
        help.push(v);
        s.pop();
    }
    else{
        while(help.top()<=v){
            auto temp = help.top();
            help.pop();
            s.push(temp);
        }
        help.push(v);
    }

}
```
### 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列`1,2,3,4,5`是某栈的压入顺序，序列`4,5,3,2,1`是该压栈序列对应的一个弹出序列,`4,3,5,1,2`就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

**思路**  设压入序列是`pushV`,弹出序列是`popV`,从`pushV`中找到第一个和`popV`首元素相同的元素，假设是`a`,可以知道`a`必定是压入栈之后就被弹出来了。假设用一个临时的栈`s1`保`a`之前入栈的元素和`a`,那么此时`popV`和`s1`的栈顶元素应该是一致的。 接下来，遍历`popV`中的元素，与`s1`中的元素比较，如果相等，弹出该元素,`popV`的指针加1,继续下一次比较.如果不相等,就把`pushV`中`a`后面的元素继续入栈到`s1`,`popV`的指针不动，继续比较`popV`和当前`s1`的栈顶元素是不是相等。按照这样的规则处理，直到指针走到尽头，看`s1`中的元素是否非空，如果是空的，`popV`就是`pushV`的弹出序列。

```c
bool IsPopOrder(vector<int> pushV,vector<int> popV) {
       if(pushV.size() != popV.size()) return false;
       stack<int> s1;
       //找到第一个相同的元素
       size_t i = 0;
       while(pushV[i] != popV[0] && i<pushV.size()){
           s1.push(pushV[i]);
           i++;
       }
       s1.push(pushV[i]); //此时是s1栈顶和popV的第一个元素相同；
       i++;//i指向下一元素；
       for(size_t j=0;j<=popV.size();j++){
           if( j<popV.size() && popV[j] == s1.top() ){
              s1.pop();
           }else if(i<pushV.size()){
               s1.push(pushV[i]);
               j--;
               i++;
           }else if( j== popV.size() && s1.empty()){
               return true;
           }else{
               return false;
           }
       }
       return false;
   }
```
