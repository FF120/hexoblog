---
title: 算法专题_链表2
toc: true

tags:
  - ACM
date: 2017-07-26 18:46:20
---

【翻转链表】【删除重复元素】【判断有环】【合并链表】【复杂链表的复制】【...】

<!-- more -->

### [反转链表](http://lintcode.com/zh-cn/problem/reverse-linked-list/)

```c
/**
* 1. 如果是空链表，或者只有一个元素，直接返回头节点
* 2. pre --> head, iter --> head->next, pre->next = nullptr;
* pre 指向第一个元素，iter指向第二个节点(可能为空，为空说明到达了链表的结尾)，断开前两个节点之间的连接
* 3. while(iter!=nullptr) 转换pre iter指向的节点的连接。
* 首先用一个新的节点变量存储iter的下一个节点信息，node = iter->next
* iter->next = pre;
* pre = iter;
* iter = node;
*/
ListNode *reverse(ListNode *head){
    if(!head || !head->next) return head;
    ListNode* pre = head;
    ListNode* iter = head->next;
    pre->next = nullptr;
    while(iter){
        ListNode* next = iter->next;
        iter->next = pre;
        pre = iter;
        iter = next;
    }
    return pre;
}
```

### [局部翻转链表](http://lintcode.com/zh-cn/problem/reverse-linked-list-ii/)

```c
ListNode * reverseBetween(ListNode * head, int m, int n) {
    // 翻转从m到n之间的链表，连接在原来的链表上
    if(!head || !head->next || m <=0 || n <=0 || m >= n) return head;
    // 1. 增加一个头节点
    ListNode* g = new ListNode(0);
    g->next = head;
    head = g;

    ListNode* pre_start = head;
    ListNode* end_next = head;
    // 1. 找到第m个节点的前一个节点
    m = m - 1;
    while(pre_start && m--){
        pre_start = pre_start->next;
    }
    // 2. 找到第n个节点的后一个节点
    n = n + 1;
    while(end_next && n--){
        end_next = end_next->next;
    }

    if(m >0 || n>0) return head;

    // 3. 翻转m 到 n之间的链表
    ListNode* stmp = pre_start->next;
    ListNode* start = pre_start->next;
    ListNode* end = start->next;
    start->next = nullptr;
    while(end != end_next){
        ListNode* tmp = end->next;
        end->next = start;
        start = end;
        end = tmp;
    }
    // 4. 连接到原来的链表上
    pre_start->next = start;
    stmp->next = end;
    return head->next;
}
```

### [K组翻转链表](http://lintcode.com/zh-cn/problem/reverse-nodes-in-k-group/)

```c
ListNode *reverseKGroup(ListNode *head, int k) {
    // 每K个节点翻转一下链表
    if(!head || k <= 1) return head;
    // 增加一个头节点
    ListNode* g = new ListNode(0);
    g->next = head;
    head = g;

    ListNode* pre_start = head;
    ListNode* end_next = head;
    while(true){
        // 1. 找到第K个节点的下一个节点
        int cc = k + 1;
        while(end_next && cc--){
            end_next = end_next->next;
        }
        
        if(cc > 0) break; // 不足K个节点的部分不翻转
        
        // 2. 翻转K个节点
        ListNode* start = pre_start->next;
        ListNode* stmp = pre_start->next;
        ListNode* end = start->next;
        while(end && end != end_next){
            ListNode* tmp = end->next;
            end->next = start;
            start = end;
            end = tmp;
        }
        //3. 连接到原来的链表
        pre_start->next = start;
        stmp->next = end_next;

        //4. 维护pre_start, end_start在正确的位置上，执行下一次循环
        pre_start = end_next = stmp;
    }
    return head->next;
}
```

### [删除链表中倒数第n个节点](http://lintcode.com/zh-cn/problem/remove-nth-node-from-end-of-list/#)

使用两个指针，遍历一遍链表就能解决；

```c
/**
*  删除倒数第K个节点，我们需要找到倒数第K+1个节点
* 1. 设链表的长度是n, 1<=k<=n
* 2. 用两个间隔是K的指针，当第二个指针指向链表的null元素的时候，第一个指针指向导数第K+1个节点
* 3. 在链表的第一个节点的前面添加一个节点guard; first --> guard
* 4. second = guard->next; second向后移动K步，如果在K大于0的时候循环结束，表示没有导数第K个元素，返回原来的链表
* 循环退出的时候，first 和second间隔K个节点，此时同时移动first second, second到达链表结尾的时候，first指向导数第K+1个节点，
* 5. 删除节点： first->next = first->next->next;
*/
ListNode *removeNthFromEnd(ListNode *head, int n) {
    ListNode *g = new ListNode(0);
    g->next = head;
    head = g;
    ListNode* first = g;
    ListNode* second = g->next;
    while(second && n--){
        second = second -> next;
    }
    if(n > 0) return head->next; // n超过了链表的长度
    while(second){
        first = first->next;
        second = second->next;
    }
    first->next = first->next->next;
    return head->next;
}
```

### [判断链表是否有环](http://lintcode.com/zh-cn/problem/linked-list-cycle/)

```c
/**
* 1. head=null 或者 head->next = null，肯定无环
* 2. first = second = head;
* 3. do{}while();  first == second时候循环结束，说明链表有环；
* 4. 一旦找到环，把first = head, 然后一步一步的移动first second, 再次相遇的地方就是环的入口。
*/
bool hasCycle(ListNode *head) {
    if(!head || !head->next) return false;
    ListNode* first = head;
    ListNode* second = head;
    do{
        first = first->next;
        second = second->next->next;
    }while(first && second && second->next && second != first);

    if(!first || !second || !second->next) return false;
    // 找到入口节点
    first = head;
    while(first != second){
        first = first->next;
        second = second->next;
    }
    return true;
}
```

### [判断链表是否相交](http://lintcode.com/zh-cn/problem/intersection-of-two-linked-lists/)

两种思路： 一种先遍历一遍两个链表，得出每个链表的长度，将尾部对其，判断

另外一种，将一个链表的尾部连接到其头结点，从另外一个链表的头节点开始遍历，转换成判断链表是否有环的问题。

```c
bool hasCycle(ListNode *head,ListNode* &enter) {
    if(!head || !head->next || !head->next->next) return false;
    ListNode* first = head;
    ListNode* second = head;
    do{
        first = first->next;
        second = second->next->next;
    }while(first && second && second->next && first != second);
    if(!first || !second || !second->next) return false;
    // 找到入口节点
    first = head;
    while(first != second){
        first = first->next;
        second = second->next;
    }
    enter = first;
    return true;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if(!headA || !headB) return nullptr;
    ListNode* iterA = headA;
    while(iterA->next !=nullptr){
        iterA = iterA->next;
    }
    iterA->next = headA;
    ListNode* enter;
    hasCycle(headB,enter);
    return enter;
}
```

### [合并K个排序的链表](http://lintcode.com/zh-cn/problem/merge-k-sorted-lists/)

1. 第一方法，用一个数组保存K个链表的头指针，每次都比较K个元素的大小，选出最小的元素，链接到新的链表的结尾，然后该链表向下一个节点移动一位。

2. 利用上面的思路，每次都要从K个元素中选择最小的元素，可以使用一个K个元素的最小堆实现，降低时间复杂度。

3. 利用归并的思想，两两合并；

```c
// 第一种方法

// 每次找出最小值的方式
 ListNode *mergeKLists(vector<ListNode *> &lists) {
    // write your code here
    ListNode* guard = new ListNode(0);
    ListNode* head = guard;

    // current 指向K个链表的头指针
    vector<ListNode*> currents;
    for(auto link : lists){
        if(link != nullptr)
            currents.push_back(link);
    }

    int index = 0;
    while(!currents.empty()){
        int minV = INT32_MAX;
        for(int i = 0; i < currents.size();++i){
            // 去掉已经为空的链表
            if(currents[i] == nullptr){
                currents.erase(currents.begin()+i);
            }
            if(currents[i]->val < minV){
                index = i;
                minV = currents[i]->val;
            }
        }
        guard->next = currents[index];
        guard = guard->next;
        if(currents[index]->next != nullptr)
            currents[index] = currents[index]->next;
        else
            currents.erase(currents.begin()+index);
    }
    return head->next;
}
```

第二种方法， 最小堆

```c
// 第二种方法， 最小堆
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

 第三种方法,归并排序的方式

```c
// 第三种方法,归并排序的方式
// 通过调用合并两个排序链表的方式实现
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
    // write your code here
    ListNode* guard = new ListNode(0);
    ListNode* head = guard;
    while(l1 && l2){
        if(l1->val > l2->val){
            guard->next = l2;
            guard = guard->next;
            l2 = l2->next;
        }else{
            guard->next = l1;
            guard = guard->next;
            l1 = l1->next;
        }
    }
    if(l1){
        guard->next = l1;
    }
    if(l2){
        guard->next = l2;
    }
    return head->next;
}

ListNode *mergeKLists(const vector<ListNode *> &lists) {
    if(lists.empty()) return nullptr;
    if(lists.size() == 1) return lists[0];
    if(lists.size() == 2) return mergeTwoLists(lists[0],lists[1]);

    size_t n = lists.size();
    vector<ListNode*> left((n+1)/2),right(n-(n+1)/2);
    copy(lists.begin(),lists.begin()+(n+1)/2,left.begin());
    copy(lists.begin()+(n+1)/2,lists.end(),right.begin());
    return mergeTwoLists(mergeKLists(left),mergeKLists(right));
}
```

上面采用的是自顶向下的递归归并的方式，下面采用非递归的自底向上的方法来实现

```c
// 简单的实现一个数据的归并求和
// 自底向上归并
int solve(vector<int> &v){
    size_t end = v.size() - 1;
    if(end < 0) throw "";
    if(end == 0) return v[0];
    while(end > 0){
        int index = 0;
        for(int i=0;i<=end;i+=2){
            if(i+1 <= end){
                v[index] = add(v[i],v[i+1]);
                index++;
            }else{
                v[index] = v[i];
                index++;
            }
        }
        end = index - 1;
    }
    return v[0];
}
```

```c
ListNode *mergeKLists(vector<ListNode *> &lists) {
    
    if(lists.empty()) return nullptr;
    if(lists.size() == 1) return lists[0];
    if(lists.size() == 2) return mergeTwoLists(lists[0],lists[1]);

    size_t end = lists.size() - 1;
    while(end > 0){
        int index = 0;
        for(int i=0;i<=end;i+=2){
            if(i+1 <= end){
                lists[index] = mergeTwoLists(lists[i],lists[i+1]);
                index++;
            }else{
                lists[index] = lists[i];
                index++;
            }
        }
        end = index - 1;
    }
    return lists[0];
}
```

### [带随机指针链表的复制|复杂链表的复制](http://lintcode.com/zh-cn/problem/copy-list-with-random-pointer/)

使用O（1）的额外空间完成链表的复制，基本思路是把拷贝的新节点链接在原来节点的后面，设置完随机指针之后再断开与原来旧链表的连接。

```c
RandomListNode *copyRandomList(RandomListNode *head) {
    // write your code here
    if(!head) return head;
    RandomListNode* iter = head;
    // 1. 复制每个节点连接在原来节点的后面
    while(iter){
        RandomListNode* node = new RandomListNode(iter->label);
        node->next = iter->next;
        iter->next = node;
        iter = node->next;
    }
    // 2. 设置随机指针
    RandomListNode* newHead = head->next;
    head->next->random = head->random==nullptr ? nullptr : head->random->next;
    iter = newHead->next;
    while(iter){
        RandomListNode* next = iter->next->next;
        iter->next->random = iter->random == nullptr ? nullptr : iter->random->next;
        iter = next;
    }
    // 3. 断开和旧链表的连接
    iter = newHead;
    while(iter->next){
        iter->next = iter->next->next;
        iter = iter->next;
    }
    return newHead;
}
```

### [删除链表元素](http://lintcode.com/zh-cn/problem/remove-linked-list-elements/)

删除链表节点为某个值的所有元素。(某个值可能有多个，也可能全部都是)

```c
ListNode *removeElements(ListNode *head, int val) {
       // Write your code here
       ListNode *guard=new ListNode(0);
       guard->next = head;
       head = guard;

       ListNode *pre = guard;
       ListNode *tmp = guard->next;
       while(tmp){
           if(tmp->val == val){
               pre->next = tmp->next;
               tmp = tmp->next;
           }else{
               pre = tmp;
               tmp = tmp -> next;
           }

       }
       return head->next;
   }
```

不使用哨兵的方法，在要求不使用额外空间的时候可以用这种方法。

```c
ListNode *removeElements(ListNode *head, int val) {
       // Write your code here
       // 删除链表中等于给定值val的所有节点
       while(head && head->val == val){
           head = head->next;
       }
       if(head == nullptr) return head;
       ListNode *pre = head;
       ListNode *c = head->next; // 头结点已经不可能==val了，因为前面已经删除了
       while(c){
           if(c->val == val){
               pre->next = c->next;
               c->next= nullptr;
               c = pre->next;
           }else{
               pre = c;
               c = c->next;
           }
       }
       return head;
}
```

### [在O(1)时间复杂度删除链表节点](http://lintcode.com/zh-cn/problem/delete-node-in-the-middle-of-singly-linked-list/)

思路：给出要删除的节点，是不可能在O(1)的时间内删除该节点的，因为想要删除该节点，必须找到它前面的节点，这个操作只能从头遍历(在没有指向前面的指针的情况下)，复杂度不可能是O(1). 但是可以把要删除的节点后面的节点的值赋给要删除的节点，再删除要删除的节点后面的节点，这样'看起来'好像删除了当前的节点。实际上在计算机内存中真正的变化是要删除的节点的值变成了它后面节点的值.

```c
void deleteNode(ListNode *node) {
        // write your code here
        node->val = node->next->val;
        node->next = node->next->next;
}
```

### [删除排序链表中的重复元素](http://lintcode.com/zh-cn/problem/remove-duplicates-from-sorted-list/)

给定一个排序链表，删除所有重复的元素每个重复元素只留下一个。

遇到不同的元素的时候再删除。

```c
ListNode *deleteDuplicates(ListNode *head) {
        // write your code here
        if(head == nullptr || head->next == nullptr) return head;
        ListNode *pre = head;
        ListNode *current = head->next;
        while(current){
            if(pre->val == current->val){
                current = current->next;
            }else{
                pre->next = current;
                pre = current;
                current = current->next;
            }

        }
        // 链表结尾有重复元素的情况
        pre->next = current;
        return head;
}
```

一旦遇到相同的元素就删除。

```c
ListNode *deleteDuplicates(ListNode *head) {
        // 给定一个排序链表，删除所有重复的元素每个元素只留下一个。
        if(head == nullptr || head->next == nullptr) return head;
        if(head->next->next == nullptr && head->val == head->next->val) return head->next;
        ListNode *pre = head;
        ListNode *c = head->next;
        while(c){
            if(pre->val == c->val){
                pre->next = c->next;
                c = c->next;
            }else{
                pre = pre->next;
                c = c->next;
            }
        }
        return head;
}
```
### [删除排序链表中的重复元素2](http://lintcode.com/zh-cn/problem/remove-duplicates-from-sorted-list-ii/)

给定一个排序链表，删除所有重复的元素只留下原链表中没有重复的元素。

```c
ListNode * deleteDuplicates(ListNode *head) {
       //给定一个排序链表，删除所有重复的元素只留下原链表中没有重复的元素。
       ListNode *guard = new ListNode(0);
       guard->next = head;
       head = guard;

       ListNode *pre = guard;
       ListNode *start = guard->next;
       ListNode *end = guard->next;
       while(end){
           while(end && start->val == end->val){
               end = end->next;
           }
           if(start->next == end){
               pre = start;
               start = end;
           }else{
               pre->next = end;
               start = end;
           }
       }
       return head->next;
}
```

不使用哨兵节点的方案

```c
ListNode * deleteDuplicates(ListNode *head) {
    //给定一个排序链表，删除所有重复的元素只留下原链表中没有重复的元素。
    ListNode *pre = head;
    ListNode *start = head;
    ListNode *end = head;
    while(end){
        while(end && start->val == end->val){
            end = end->next;
        }
        if(start->next == end){
            pre = start;
            start = end;
        }else{
            if(pre == start && pre == head){
                head = end;
                pre = end;
            }
            else    pre->next = end;
            start = end;
        }
    }
    return head;
}
```

### 删除单向链表中倒数第K个节点

例如：`head-->1-->2-->3-->4-->5-->6`  K=2
返回：`head-->1-->2-->3-->4-->6`

思路一： 先遍历一遍链表，统计链表有多少个元素。然后就可以知道倒数第K个元素就是
正数第n-k+1个元素，从head开始移动指针，移动n-k次就是要删除的节点。
删除一个节点，只需要把原来指向这个元素的指针指向它后面的元素就可以了。所以我们要做的
操作就是从head开始，移动n-k-1次，找到要删除的节点前面的节点a,然后执行a->next = a->next->next

```c
ListNode *removeNthFromEnd(ListNode *head, int n) {
    // write your code here
    int len = 0;
    ListNode *tmp = head;
    while(tmp){
        len ++;
        tmp = tmp->next;
    }
    if(len < n) return nullptr;
    if(len == n) return head->next;

    tmp = head;
    int count = len + 1 - n  ;// 正数第count个节点
    count = count - 2;
    while(count--){
        tmp = tmp->next;
    }
    tmp->next = tmp->next->next;
    return head;
}
```

思路二：使用两个间隔K个元素的指针遍历一遍链表，当前面的指针走到尾节点的时候，后面的节点指向的就是倒数第K个节点。

```c
ListNode *removeNthFromEnd(ListNode *head, int n) {
    // write your code here
    if(head == nullptr) return nullptr;
    ListNode *first = head;
    ListNode *second = head;
    int count = n + 1;
    while(first && count--){
        first = first->next;
    }
    if(first == nullptr) return head->next; // 删除第一个节点
    if(count > 0) return nullptr; // 没有导数第n个节点
    while(first){
        first = first->next;
        second = second->next;
    }
    second->next = second->next->next;
    return head;
}
```

### 查找链表的中间节点

思路： 设置两个指针同时指向头部，一个指针每次移动两步，一个指针每次移动一步，这样在快的指针走到尽头的时候，慢的指针恰好走到了中部。

```c
/**
 * 找到链表的中间节点并返回
 * @param head
 * @return
 */
ListNode* findMiddle(ListNode* head){
    if(head == nullptr || head->next == nullptr) return head;
    ListNode *first = head, *second = head;
    while(second->next && second->next->next){
        first = first->next;
        second = second->next->next;
    }
    return first;
}
```

### 插入排序（使用单链表实现）

```c
ListNode *insertionSortList(ListNode *head) {
    // 使用链表实现插入排序
    if(head == nullptr || head->next == nullptr) return head;

    ListNode *guard = new ListNode(INT32_MIN);
    guard->next = head;
    head = guard;

    ListNode *sorted = guard;
    ListNode *ends = guard->next;
    ListNode *unsorted = ends->next;
    ends->next = nullptr; // 链表断开
    while(unsorted){
        // 找到插入的位置
        while(sorted != ends){
            if(unsorted->val >= sorted->val && unsorted->val < sorted->next->val){
                // 插入到iter1后面
                ListNode *tmp = unsorted;
                unsorted = unsorted->next;
                tmp->next = sorted->next;
                sorted->next = tmp;
                sorted = guard;
                break;
            }
            sorted = sorted->next;
        }
        if(sorted == ends && unsorted->val >= ends->val){
            ends->next = unsorted;
            unsorted = unsorted->next;
            ends = ends->next;
            ends->next = nullptr;
            sorted = guard;
        }

    }
    return head->next;
}
```

### 快速排序(单链表实现)

快速排序一般的思想是首尾两个指针相向运动，但是单链表没有向前的指针，这要求我们换一种思路。如何从前往后遍历找到支点的位置。假设支点的值是value. 我们需要遍历一遍链表，使得所有小于value的值都在前面，所有大于value的值都在后面，这个时候就找到的value应该在的位置。

```c
// 单链表实现快排

```

### [交换链表当中两个节点](http://lintcode.com/zh-cn/problem/swap-two-nodes-in-linked-list/)

给你一个链表以及两个权值v1和v2，交换链表中权值为v1和v2的这两个节点。保证链表中节点权值各不相同，如果没有找到对应节点，那么什么也不用做。



```c
ListNode* swapNodes(ListNode* head, int v1, int v2) {
    // Write your code here
    if(head==nullptr) return head;

    // add a guard
    ListNode *guard = new ListNode(0);
    guard->next = head;
    head = guard;

    //search the value
    ListNode *pre_v1=nullptr;
    ListNode *pre_v2=nullptr;
    ListNode *pre_tmp=guard;
    ListNode *tmp=pre_tmp->next;
    while(tmp){
        if(tmp->val==v1){
            pre_v1 = pre_tmp;
        }
        if(tmp->val==v2){
            pre_v2 = pre_tmp;
        }
        pre_tmp = tmp;
        tmp = tmp->next;
    }

    // not find v1 or v2
    if(pre_v1==nullptr || pre_v2==nullptr){
        return head->next;
    }

    // swap the two nodes
    ListNode *v1s = pre_v1->next;
    ListNode *v2s = pre_v2->next;
    ListNode *next_v1 = v1s->next;
    ListNode *next_v2 = v2s->next;

    if(pre_v1 == v2s){
        pre_v2->next = v1s;
        v1s->next = v2s;
        v2s->next = next_v1;
    }else if(pre_v2 == v1s){
        pre_v1->next = v2s;
        v2s->next = v1s;
        v1s->next = next_v2;
    }else{
        v1s->next = next_v2;
        v2s->next = next_v1;
        pre_v1->next = v2s;
        pre_v2->next = v1s;
    }

    return head->next;
}
```

### [两两交换链表中的节点](http://lintcode.com/zh-cn/problem/swap-nodes-in-pairs/)

给一个链表，两两交换其中的节点，然后返回交换后的链表。

```c
ListNode* swapPairs(ListNode* head) {
        // Write your code here
        if(head==nullptr || head->next==nullptr) return head;

        ListNode *guard = new ListNode(0);
        guard->next = head;
        head = guard;

        ListNode *pres1 = guard;
        ListNode *s1 = guard->next;
        ListNode *s2 = guard->next->next;
        while(s1!=nullptr && s2!=nullptr){
            pres1->next = s2;
            s1->next = s2->next;
            s2->next = s1;
            if(s1->next==nullptr || s1->next->next==nullptr){
                return head->next;
            }else{
                pres1 = s1;
                s1 = s1->next;
                s2 = s2->next->next->next;
            }
        }
        return head->next;
    }
```

### [旋转链表](http://lintcode.com/zh-cn/problem/rotate-list/)

```c
ListNode *rotateRight(ListNode *head, int k) {
    //给定一个链表，旋转链表，使得每个节点向右移动k个位置，其中k是一个非负数
    if(head == nullptr || head->next == nullptr || k <= 0) return head;
    ListNode *end = head;
    int len = 1;
    while(end->next){
        end = end->next;
        len++;
    }
    k = k % len;
    if(k == 0) return head;
    int index = (len - 1 - k) % len;
    ListNode *tmp = head;
    while(tmp && index --){
        tmp = tmp->next;
    }
    if(index == -1){
        end->next = head;
        head = tmp->next;
        tmp->next = nullptr;
    }
    return head;
}
```

### [首尾交错重排链表](http://lintcode.com/zh-cn/problem/reorder-list/)

给定一个单链表L: L0→L1→…→Ln-1→Ln,重新排列后为：L0→Ln→L1→Ln-1→L2→Ln-2→…必须在不改变节点值的情况下进行原地操作。


```c
/**
 * 找到链表的中间节点并返回
 * @param head
 * @return
 */
ListNode* findMiddle(ListNode* head){
    if(head == nullptr || head->next == nullptr) return head;
    ListNode *first = head, *second = head;
    while(second->next && second->next->next){
        first = first->next;
        second = second->next->next;
    }
    return first;
}
/**
 * 反转链表
 * @param head
 * @return
 */
ListNode *reverse(ListNode *head) {
    // write your code here
    if(head==nullptr || head->next==nullptr) return head;
    ListNode *iter =head;
    ListNode *next_iter = head->next;
    iter->next=nullptr;
    while(next_iter){
        ListNode *tmp = next_iter->next;
        next_iter->next = iter;
        iter = next_iter;
        next_iter = tmp;
    }
    head = iter;
    return head;
}
/**
 * 交叉合并两个链表
 * @param l1
 * @param l2
 * @return
 */
ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
    // 如果其中一个链表为空，直接返回另外一个链表就是结果
    if(l1 == nullptr) return l2;
    if(l2 == nullptr) return l1;
    ListNode *head = l1;
    l1 = l1->next;
    ListNode *iter = head;
    bool odd = false;
    while(l1 != nullptr || l2 != nullptr){
        if(!odd){
            if(l2){
                iter->next = l2;
                l2 = l2->next;
                iter = iter->next;
            }
        }
        if(odd){
            if(l1){
                iter->next = l1;
                l1 = l1->next;
                iter = iter->next;
            }

        }
        odd = !odd;
    }
    return head;
}

ListNode* reorderList(ListNode *head) {
    if(head == nullptr || head->next == nullptr) return head;
    // 给定一个单链表L: L0→L1→…→Ln-1→Ln,重新排列后为：L0→Ln→L1→Ln-1→L2→Ln-2→…必须在不改变节点值的情况下进行原地操作
    // 找到中间节点， 反转后半段，然后交叉拼接起来，可以确保不适用额外的空间
    ListNode* middle = findMiddle(head);
    ListNode* head2 = reverse(middle->next);
    middle->next = nullptr;
    head = mergeTwoLists(head,head2);
    return head;
}
```

### [回文链表](http://lintcode.com/zh-cn/problem/palindrome-linked-list/)

判断一个链表中的值是否是回文

找到中间节点，把后面的链表反转，然后与前面的逐个节点比较，如果完全一样或者只有最后一个节点不一样，则是回文的。

```c
bool isPalindrome(ListNode* head) {
    // Write your code here
    if(head == nullptr) return true;
    ListNode *middle = findMiddle(head);
    ListNode *head2 = reverse(middle->next);
    middle->next = nullptr;
    ListNode *iter1 = head, *iter2 = head2;
    while(iter1 && iter2){
        if(iter1->val != iter2->val) return false;
        iter1 = iter1->next;
        iter2 = iter2->next;
    }
    if(iter1 && iter1->next == nullptr){
        return true;
    }
    if(iter2 && iter2->next == nullptr){
        return true;
    }
    if(iter1 == nullptr && iter2 == nullptr){
        return true;
    }
    return false;
}
```

### [链表求和](http://lintcode.com/zh-cn/problem/add-two-numbers/)

你有两个用链表代表的整数，其中每个节点包含一个数字。数字存储按照在原来整数中相反的顺序，使得第一个数字位于链表的开头。写出一个函数将两个整数相加，用链表形式返回和。

```c
ListNode *addLists(ListNode *l1, ListNode *l2) {
    if(l1 == nullptr) return l2;
    if(l2 == nullptr) return l1;
    // 链表求和
    int carry = 0;
    ListNode *head = new ListNode(0);
    ListNode *iter = head;
    int number;
    while(l1 != nullptr && l2 != nullptr){
        number = (l1->val + l2->val + carry ) % 10 ;
        carry = (l1->val + l2->val + carry ) / 10 ;
        ListNode *node = new ListNode(number);
        iter->next = node;
        iter = iter->next;
        l1 = l1->next;
        l2 = l2->next;
    }
    while(l1){
        number = (l1->val + carry) % 10;
        carry = (l1->val + carry) / 10;
        ListNode *node = new ListNode(number);
        iter->next = node;
        iter = iter->next;
        l1 = l1->next;

    }
    while(l2){
        number = (l2->val + carry) % 10;
        carry = (l2->val + carry) / 10;
        ListNode *node = new ListNode(number);
        iter->next = node;
        iter = iter->next;
        l2 = l2->next;
    }
    if(carry){
        ListNode *node = new ListNode(carry);
        iter->next = node;
    }
    return head->next;
}
```

### [链表求和 II](http://lintcode.com/zh-cn/problem/add-two-numbers-ii/#)

假定用一个链表表示两个数，其中每个节点仅包含一个数字。假设这两个数的数字顺序排列，请设计一种方法将两个数相加，并将其结果表现为链表的形式。

先翻转链表，求和，在翻转回去。

```c
ListNode *addLists2(ListNode *l1, ListNode *l2) {
    return reverse(addLists(reverse(l1),reverse(l2)));
}
```