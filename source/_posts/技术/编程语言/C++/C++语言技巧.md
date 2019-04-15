---
title: C++语言技巧
toc: true
tags:
  - C++
date: 2017-04-07 11:29:34
---

C++有关的概念和常用的代码。主要包括内存模型(堆内存，栈内存)，虚函数，虚表，const 的用法，include，sizeof, 构造函数等内容。

<!--more-->

## 基础概念

### include (文件包含)

**`include <>` 和 `include ""` 有什么区别？**

`<>`和`""`的区别是系统在搜索头文件的时候顺序不同，`<>`在搜索的时候首先从系统目录开始搜索，然后搜索path环境变量下面，不搜索当前的目录； `""`首先搜索当前目录，然后搜索系统和path目录，所以自己写的文件用双引号，系统自带的库用`<>`,主要是为了搜索快。

**如何避免头文件被重复包含**

按照如下的格式书写头文件：

```c
#ifndef _HEADERNAME_H
#define _HEADERNAME_H

...//(头文件内容)

#endif
```

这样头文件在第一次被包含的时候，_HEADERNAME_H 没有被定义，执行定义_HEADERNAME_H的动作并包含头文件的内容，
第二次包含的时候，_HEADERNAME_H 已经被定义，就不会包含后面的内容；`_HEADERNAME_H`是自定义的名称，需要为每一个头文件起一个不一样的名称，这样才能达到效果。

### 常量(const)

**常量的定义方式和异同？**

有两种定义方式： `define PI 3.14` 和 `const double PI = 3.14`.

`define` 定义的常量是在编译之前的预处理阶段执行的简单的字符串替换，就是把代码中所有出现`PI`的地方替换成`3.14`,不执行语法和类型的检查。 `const`定义的常量是在编译阶段处理的，有类型的检查和语法的检查，更安全。 如果使用`define`定义复杂的常量表达式，需要特别注意括号的使用。

**顶层const和底层const**

`const`既可以修饰普通的变量(整形，字符串等)也可以修饰指针，假设一个指针`p`指向一个变量`a`, 如果指针是常量(`p`中存储的地址不能改变) 叫做顶层const, 如果`a`是常量(`a`中存储的数值不能改变)，叫做底层const.

顶层const的定义： `int *const p = &a;`  p 的值不能改变，但是可以通过p改变它指向的a的值；
底层const的定义： `const int *p = &a;`  p 的值可以改变，但是不能通过p改变它指向的a的值。

既是顶层也是底层：`const int *const p = &a;` p的值不能改变，也不能通过p改变a的值；

**常量表达式**

值不会改变并且在编译的时候可以确定值的表达式就是常量表达式，可以使用`constexpr`声明常量表达式，这样编译器会自己判断表达式是否是常量表达式，如果不是，就会报错。`constexpr int m = 20 + 90;`

`constexpr` 还可以用来修饰函数的返回值，这是一种常量函数，要保证在编译的时候就能得到结果。`constexpr int getIntSize(){return 4;}`这样一个函数在编译的时候就可以确定其返回值，所以该函数可以用来初始化常量。 需要注意的是，并不一定加了constexpr的函数就一定返回常量，也可以返回非常量，如果用返回非常量的`constexpr`函数初始化常量，编译器会报错。

### 虚函数

对于非 虚函数的调用，在编译的时候确定调用哪一个，例如函数的重载，通过函数参数的类型，个数就可以确定调用哪一个；
对于虚函数的调用，直到运行的时候才能确定应该调用哪一个函数。 当使用基类的引用调用基类的虚函数的时候，编译器是无法确定到底调用哪一个函数的，需要等到运行时，了解基类的指针或者引用具体绑定到了哪一个子类上，才能确定下来（调用该子类自己实现的函数版本).

1. 虚函数 不代表 不被实现， 纯虚函数才是不被实现的函数；
2. 虚函数的目的是允许用基类的指针调用子类的这个函数；（在基类中声明为虚函数就可以，子类中不用带virtual）
3. 纯虚函数的目的是定义一个接口，规定所有继承该类的的子类必须实现这个函数；
4. 包含纯虚函数的类是抽象类，不能实例化，不能创建类的实例。

虚析构函数的作用是在销毁对象的时候首先销毁基类指针指向的子类对象的实例，再执行基类对象的析构函数，如果子类不定义程虚析构函数，则不会执行子类的析构函数，容易造成内存泄漏。

### 堆内存和栈内存

http://www.cnblogs.com/yyxt/archive/2015/02/02/4268304.html

`char a[] = "1234"; ` 是存储在栈上的；

`char *a = "1234";` 是存储在堆上的；

栈内存访问的速度快于堆内存，因为堆内存的指针也是存放在栈上的，需要先访问栈，然后去堆上访问，对的地址也是不连续的，这导致访问的速度下降。

**new malloc**

`new` 和 `malloc` 开辟的内存是存储在堆上的，需要自己去释放内存，否则只有在程序结束之后才有可能被操作系统回收。
`delete` 和 `free` 是用来释放内存的，delete 或调用被释放对象的析构函数，安全的释放内存，`free` 直接释放。

### 静态链接和动态连接

源文件-->预编译-->编译-->汇编--> **链接** -->可执行程序

如果在链接的时候，将源文件中用到的库函数与汇编生成的文件合并生成一个可执行文件，之后的程序仅需要这个可执行文件即可运行，这样的方式叫做静态链接； 缺点是文件可能太大，毕竟一个简单的程序也会包含很多的头文件。

如果在链接的时候，不把源文件中用到的库合并在一起，而是单独编译，在运行的时候，用到的地方在去寻找该库，这种方式叫做动态链接，动态连接可以有效的避免重复，但是可移植性就受到限制，经常遇到的运行某个程序的时候XX找不到的错误就是动态链接库需要的文件丢失后者没有在正确的路径上导致的。

参考文章：[http://www.cnblogs.com/52php/p/5681711.html](http://www.cnblogs.com/52php/p/5681711.html)



## 基本操作

### 字符串基本操作

```c
#include <string>

// 末尾添加一个字符
s.push_back('a');
// 末尾追加一个字符串
s.append("aaa");
// 任意位置插入字符

//任意位置插入字符串

//任意位置删除字符

//任意位置删除字符串

```

### 判断x是否是素数

```c
// 判断x是否是素数
bool isPrime(int x){
    int xx = (int)sqrt((double)x);
    for(int i=2;i<=xx;i++){
        if( x % i == 0) return false;
    }
    return true;
}
```

### 找出n以内，2，3，5整除的所有数字

```c
// 返回三个数字中的最小值和最小值的索引
int minThree(vector<int> v, int &index){
    if(v.size()!=3) return -1;
    int minValue = v[0];
    index = 0;
    if(minValue > v[1]){
        minValue = v[1];
        index = 1;
    }
    if(minValue > v[2]){
        minValue = v[2];
        index = 2;
    }
    return minValue;
}
// 找出n以内所有能被2，3，5其中一个或者多个整除的数字
vector<int> getNumbers(int x){
    vector<int> results={1};
    vector<int> index235 = {1,1,1};
    int index = 0;
    int minValue = minThree({2*index235[0],3*index235[1],5*index235[2]},index);
    while(minValue<=x){
        if(minValue != results[results.size()-1]){
            results.push_back(minValue);
        }
        index235[index]++;
        minValue = minThree({2*index235[0],3*index235[1],5*index235[2]},index);
    }
    return results;
}
```

### 找出第1500个只包含2或3或5为因子的数字(从1开始)

```c
// 返回三个数字中的最小值和最小值的索引
int minThree(vector<int> v, int &index){
    if(v.size()!=3) return -1;
    int minValue = v[0];
    index = 0;
    if(minValue > v[1]){
        minValue = v[1];
        index = 1;
    }
    if(minValue > v[2]){
        minValue = v[2];
        index = 2;
    }
    return minValue;
}
// 找到第K个数字（从1开始）
int getNumbers(int k){
    vector<int> index235 = {1,1,1};
    int index = 0;

    int tmp = 1;
    int count = 1;
    int minValue = 1;
    while(count < k){
        minValue = minThree({2*index235[0],3*index235[1],5*index235[2]},index);
        if(minValue != tmp){
            count++;
            tmp = minValue;
        }
        index235[index]++;
    }
    return minValue;
}
```


### 小于等于n的所有素数

```c
// 找出x以内的所有素数
vector<int> getPrimes(int x){
    vector<int> Primes;
    // 初始化 0 - x 都是素数
    vector<bool> isPrime(x+1,true);
    isPrime[0] = false; // 0 不是素数
    isPrime[1] = false; // 1 不是素数
    for(int i=2;i<=x;i++){
        // 如果i是素数，把所有i的倍数设置成不是素数
        if(isPrime[i]){
            Primes.push_back(i);
            for(int j=i*2;j<=x;j=j+i){
                isPrime[j] = false;
            }
        }
    }
    return Primes;
}
```

### 最大公约数

```c
// 最大公约数
int getY(int x,int y){
    int tmp = 0;

    while(y){
        tmp = y;
        y = x % y;
        x = tmp;
    }
    return x;
}
```

### 自定义set的比较函数

存入set的元素默认是有序的，但是默认的比较可能不能满足我们的要求，这个时候
就需要自定义比较的函数。 set的排序是使用红黑树的结构，插入删除和取出最小的
元素都比较高效。

```c
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

```c
multiset<NumBit> s; //
```

### 整数转换成字符串

```c
#include <sstream>
#include <string>
string Int_to_String(int n)
{
    ostringstream stream;
    stream<<n;  //n为int类型
    return stream.str();
}
```
### 十进制数字转换成K进制之后数位之和

```c
// 10进制数字 转换成K进制之后各个数位的数字之和
int getSum(int n,int k){
    int sum = 0;
    while(n){
        sum += n % k;
        n = n / k;
    }
    return sum;
}
```

### 十进制数字转换成K进制

```c
deque<int> Kin(int n,int k){
    deque<int> result;
    while(n/k != 0){
        result.push_front(n%k);
        n = n / k;
    }
    result.push_front(n);
    return result;
}
```
### K进制数字转换成十进制
```c
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

### 输入输出重定向

有的算法题是从接收的是从控制台的输入，而且输入还很多，这个时候如果每次调试都从控制台一次一次的输入测试数据，就会很麻烦。我们可以把要输入的数据保存在一个文本文件中，然后使用输入重定向[`freopen`](http://www.cplusplus.com/reference/cstdio/freopen/)把标准输入重定向到该文件。以输入一个m行n列的矩阵来说，首先把输入数据存储在文本文件`d:/A.in`中。
```
4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
```
然后执行下面的代码：
```c
    freopen("d:\\A.in","r",stdin);// 输入重定向
    int m,n;
    cin>>m>>n;
    vector<vector<int>> v(m,vector<int>(n));
    //读取数据
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            cin>>v[i][j];
        }
    }
    //输出读取的数据门这里输出到控制台
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            cout<<v[i][j]<<" ";
        }
        cout<<endl;
    }
```
同样，输出也可以重定向到文件，当有大量的输出或者需要保存输出结果的时候，重定向到文件是一个不错的方法。只需要在输出之前加上下面这段代码，输出就会重定向到文件，这个时候运行程序，控制台就看不到输出了。
```c
freopen("d:\\A.out","w",stdout);
```

### 格式化输入输出

C++定义了一些操纵符来控制输出流的状态，endl就是一个常用的操纵符。

**控制布尔值的格式**

`boolalpha`使得布尔值输出`true` or `false`;
`noboolalpha`使得输出变回默认的`0` or `1`.
```c
cout<<"default: "<<true<<" "<<false<<endl;
cout<<boolalpha<<"boolalpha: "<<true<<" "<<false<<noboolalpha<<endl;
```

**控制整数的输出进制**

- 八进制： `oct`
- 十六进制： `hex`
- 十进制： `dec`

**控制固定小数点位数**
```c
#include <iomanip>

cout.precision(6);
cout.setf(ios::fixed);

```

### 数据的表示范围

以下内容来源于`C++ Premier 第五版`

**整型**

包括整数，字符型，和布尔类型；这类数据在计算机的内部都是以二进制位0和1直接保存的。

```c
// 获得整形类型的表示范围， climits
    cout<<"char: "<<CHAR_MIN<<" to "<<CHAR_MAX<<endl;
    cout<<"unsinged char: "<<0<<" to "<<UCHAR_MAX<<endl;
    cout<<"int8: "<<INT8_MIN<<" to "<<INT8_MAX<<endl;
    cout<<"unsinged int8: "<<0<<" to "<<UINT8_MAX<<endl;
    cout<<"int16: "<<INT16_MIN<<" to "<<INT16_MAX<<endl;
    cout<<"unsigned int16: "<<0<<" to "<<UINT16_MAX<<endl;
    cout<<"int32: "<<INT32_MIN<<" to "<<INT32_MAX<<endl;
    cout<<"unsigned int32: "<<0<<" to "<<UINT32_MAX<<endl;
    cout<<"int64: "<<INT64_MIN<<" to "<<INT64_MAX<<endl;
    cout<<"unsigned int64: "<<0<<" to "<<UINT64_MAX<<endl;
    cout<<endl;
```

**浮点型**

在计算机内部，这种类型是把保存数据的空间分成两部分，一部分存储小数部分，一部分存储指数部分，数的实际大小是通过计算得出来的。
浮点类型由四部分组成：
- sign : 符号，正 或 负
- base(radix) : 基数(2,8,10,16)
- significand : 尾数
- exponent ： 指数

浮点类型的大小可以通过包含[`cfloat`](http://www.cplusplus.com/reference/cfloat/)查看。

```c
//获得浮点类型的表示范围  cfloat
   cout<<"float range: "<<FLT_MIN<<" to "<<FLT_MAX<<endl;
   cout<<"float significand: "<<FLT_MANT_DIG<<endl;
   cout<<"float exponent: "<<FLT_MIN_EXP <<" to "<<FLT_MAX_EXP<<endl;

   cout<<"double range: "<<DBL_MIN<<" to "<<DBL_MAX<<endl;
   cout<<"double significant: "<<DBL_MANT_DIG <<endl;
   cout<<"double exponent: "<<DBL_MIN_EXP <<" to "<<DBL_MAX_EXP<<endl;

   cout<<"long double range: "<<LDBL_MIN<<" to "<<LDBL_MAX<<endl;
   cout<<"long double significant: "<<LDBL_MANT_DIG<<endl;
   cout<<"long double exponent"<<LDBL_MIN_EXP<<" to "<<LDBL_MAX_EXP<<endl;

   cout<<"base: "<<FLT_RADIX<<endl;
   cout<<endl;
```

**获得类型所占用的字节数目**

```c
// 获得类型在内存中占的字节数
    cout<<"bool: "<<sizeof(bool)<<endl;
    cout<<"char: "<<sizeof(char)<<endl;
    cout<<"short: "<<sizeof(short)<<endl;
    cout<<"int: "<<sizeof(int)<<endl;
    cout<<"long: "<<sizeof(long)<<endl;
    cout<<"long long : "<<sizeof(long long)<<endl;
    cout<<"float: "<<sizeof(float)<<endl;
    cout<<"long double: "<<sizeof(long double)<<endl;
    cout<<endl;
```

**类型的使用准则**

- 明确知道不可能为负，使用无符号数。
- 整数运算一般使用`int`, 需要大数的时候考虑`long long`.需要小整数的时候考虑`signed char` or `unsigned char`
- 浮点运算用`double`

### 快速幂和矩阵快速幂

**整数的快速幂**

求$a^b$一般的做法是用一个循环，将a累乘b次，这样需要做b次乘法。快速幂的思想是利用了 $a^(b1+b2) = a^b1 + a^b2$ 的思想，把b表示成二进制，然后拆分开，分别求幂，再求和。举例来说：

假设要求$5^{12}$,传统的方法是12个5相乘，要做12次乘法运算。快速幂的思想是把12表示成二进制，`1100` = $2^2+2^3$,
$$
5^{12} = 5^{(2^2+2^3)} = 5^{2^2} * 5^{2^3}
$$
2的幂的计算可以由十分迅速的移位计算得到，所有原来需要12个乘法运算才能解决的计算问题，现在编程了只需要三次计算节能解决。

```c
int quickPow(int a,int b){
    int ans=1,base=a;
    while(b!=0){
        if(b&1!=0)
        　　ans*=base;
        base*=base;
        b>>=1;
　 }
    return ans;
}
```
按照上面的代码计算出来的实际上是$5^{2^2} * 5^{2^3}$, 因为我们用的是右移，每次都只判断末尾的一个二进制位，如果是1，就乘入当前的结果。每次循环（不管是不是1），base都要翻倍，因为是二进制，每移动一位就意味着乘以2.

另外一个需要注意的问题是，实际使用时需要注意数据的范围，如果int的范围不够，可以使用long long类型。

**矩阵的快速幂**

- 矩阵乘法
一个$m*n$的矩阵  乘以 一个$n*p$的矩阵，会得到一个$m*p$的矩阵。矩阵相乘的规则是：第一个矩阵的每一行乘以第二个矩阵的每一列，对应的元素相乘再相加，作为新矩阵对应位置上的元素。朴素的矩阵乘法的代码如下：
```c
typedef vector<vector<int>> matrix;
matrix MatricMul(matrix A,matrix B){
    int m = A.size();
    int n1 = A[0].size();
    int n2 = B.size();
    int p = B[0].size();
    if(n1 != n2) {cout<<"no cheng of the two matrix."<<endl;return matrix();}
    int n = n1 = n2;
    matrix C(m,vector<int>(p,0));
    for(int i=0;i<m;i++){
        for(int j=0;j<p;j++){
            for(int k=0;k<n;k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main(){
    matrix A = {{1,2,3},{4,5,6}};
    matrix B = {{1,2},{3,4},{5,6}};
    matrix C = MatricMul(A,B);
    return 0;
}
```

- $A^n$ 快速求矩阵的n次幂，注意这里A只能是方阵
矩阵的快速幂和整数的快速幂是一样的，就是重载一下*这个运算符，使得两侧是矩阵的时候，计算的是矩阵乘法。这里我们就不重载运算符了，直接使用上面定义的矩阵乘法函数`MatricMul`:
```c
matrix quickPowMatrix(matrix A,int n){
    matrix base = A;
    // 初始化成单位矩阵
    int len = A.size();
    matrix ans(len,vector<int>(len,0));
    for(int i=0;i<A.size();i++){
        ans[i][i] = 1;
    }
    while(n!=0){
        if(n&1!=0)
            ans = MatricMul(ans,base);
        base = MatricMul(base,base);
        n>>=1;
    }
    return ans;
}
```

快速幂通常用来求很大的数，这个时候虽然就算速度在可以接受的范围内，但是数据的范围早已经超过了能够表示范围，通常的方法就是mod每个大数，得到一个较小的结果。

为了减少计算的开销（计算小数的乘法要比计算大数的乘法开销小），通常利用模运算的法则：

$$
(a+b) mod c = (a mod c + b mod c) mod c ;
$$

$$
(a*b) mod c = (a mod c * b mod c) mod c ;
$$

上面的代码每一次计算之后就取模，就可以保证数据的范围不溢出，还能保证比较快的计算速度。

**使用C++的模版技术编写通用的快速幂模版**
```c
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;

template<class T, int MAXN, T MOD=-1>
class Matrix {
public:
    T m[MAXN][MAXN];
    Matrix(){}
    // 如果MOD不是-1，把num取模
    void init(T num[MAXN][MAXN]){
        for(int i = 0 ; i < MAXN ; i++)
        {
            for(int j = 0 ; j < MAXN ; j++)
            {
                m[i][j] = num[i][j];
                if (MOD!=-1)
                    m[i][j] %= MOD;
            }
        }
    }
    //矩阵乘法的实现
    friend Matrix operator*(const Matrix &m1 ,const Matrix &m2)
    {
        int i, j, k;
        Matrix ret;
        memset(ret.m, 0, sizeof(ret.m));
        for (i = 0; i < MAXN; i++) {
            for (j = 0; j < MAXN; j++)
                if ( m1.m[i][j] )
                {
                    for(k = 0 ; k < MAXN ; k++){
                        ret.m[i][k] += m1.m[i][j] * m2.m[j][k];
                        if (MOD!=-1) ret.m[i][k] %= MOD;
                    }
                }
        }
        return ret;
    }
    // 矩阵加法的实现
    friend Matrix operator+(const Matrix &m1 ,const Matrix &m2) {
        int i, j;
        Matrix ret;
        for (i = 0; i < MAXN; i++) {
            for (j = 0; j < MAXN; j++) {
                ret.m[i][j] = 0;
                ret.m[i][j] = m1.m[i][j]+m2.m[i][j];
                if (MOD!=-1)
                    ret.m[i][j] %= MOD;
            }
        }
        return ret;
    }
    //矩阵快速幂的实现
    friend Matrix operator^(const Matrix &_M , LL nx){
        Matrix ret,M(_M);
        //ret 初始化成单位矩阵
        for(int i = 0 ; i < MAXN ; i++){
            for(int j = 0 ; j < MAXN ; j++){
                if(i == j)
                    ret.m[i][j] = 1;
                else ret.m[i][j] = 0;
            }
        }
        while(nx){
            if(nx & 1)
                ret = ret * M;
            nx = nx >> 1;
            M = M * M;
        }
        return ret;
    }
};

int main(){
    int C[2][2] = {{1,2},{3,4}};
    Matrix<int,2,1000> mm;
    mm.init(C);
    auto add = mm + mm;
    auto cheng = mm * mm;
    auto mi = mm ^ 2 ;
    return 0;
}
```
### 包含一切的头文件

```c
#include <bits/stdc++.h>
```
一个文件包含了所有常用的头文件，你所有使用的函数不再需要引入相应的头文件。该头文件在ACM竞赛中经常被使用，可以减少你包含需要的头文件需要的时间。

需要注意的是，这个头文件并不是标准的，这意味着可能有的编译器不支持它。

### 暂停和计时

**暂停**

如果想要让程序暂停几秒继续执行，可以这样使用：

```c
# include <windows.h>
Sleep(2000);  // 暂停2s, 参数的单位是毫秒
```

**计算程序运行的时间**

有的时候可能要看某段程序运行需要多少时间，可以这样使用：
- 秒级计时
```c
#include <ctime>
auto start_time = time(nullptr);
//Sleep(3000);
// ... 代码块
auto end_time = time(nullptr);
cout<<end_time - start_time<<endl;
// 输出的是程序运行的秒数。
```
- 毫秒级计时


```c
// 获取毫秒级别的时间差
auto start_time = clock();
//Sleep(3000);
auto end_time = clock();
cout<<end_time - start_time<<endl;
```

### 返回一个无序数组排序之后的下标，不动原来的数组

例如 a = [3,5,2,4,1] , 从小到大排序之后应该是[1,2,3,4,5], 原来在a中的下标是[4,2,0,3,1],我们的目标就是输入a，返回[4,2,0,3,1]
```c
#include <iostream>
#include <algorithm>

using namespace std;

vector<int> getOrderIndex(vector<int> &a){
    vector<int> order(a.size(),0);
    for(int i=0;i<a.size();i++){
        order[i] = i;
    }
    sort(order.begin(), order.begin() + a.size(), [a](const int& x, const int& y)->bool { return a[x] < a[y];});
    return order;
}
```
order中就是我们想要的结果。

### 读取数量不定的若干个整数

```c
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;
/**
 * 读取一行整数，返回数组
 * @param s
 * @return
 */
vector<int> getInt(string &s)
{
    getline(cin,s);
    istringstream iss(s);
    vector<int> v;
    int num;
    while(iss >> num){
        v.push_back(num);
    }
    return v;
}
```

### 字符串转换成整数

```c
/**
 * 用空格分割的字符串转换成整数
 * @param s 
 * @return 
 */
vector<int> string2int(string &s){
    istringstream in(s);
    vector<int> v;
    int num;
    while(in >> num){
        v.push_back(num);
    }
    return v;
}
```

### 输入挂

当纯数字的输入规模超过$10^6$时，可以考虑使用输入挂，比系统自带的cin快很多。

```c
inline void q_read(int &num)
{
    char ch; int f = 1;
    while(true)
    {
        ch = getchar();
        if(ch == '-') f = -1;
        if(isdigit(ch))
        {
            num = ch - '0';
            break;
        }
    }
    while(ch = getchar(), isdigit(ch)) num = num*10+ch-'0';
    num *= f;
}
```

还可以在开始的时候加入 `ios::sync_with_stdio(false);`, 它的作用是去掉cin额外的检查开销，达到和scanf相似的输入效率；
