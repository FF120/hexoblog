---
title: SQL
toc: true

tags:
  - SQL
date: 2017-11-02 14:15:52
---

基于MySQL的语法说明数据库查询的一些操作。

在`test`数据库中创建一个`user`表，执行一些查询操作。

```sql
6	Li Xiao	10	2017-11-01
7	Li Xiao	20	2017-11-02
8	Li Xiao	10	2017-11-03
9	Zhao Hi	20	2016-10-11
10	Zhao Hi	20	2016-10-12
11	Zhao Hi	20	2016-10-13
12	Kao Ha	10	2016-03-01
13	Kao Ha	20	2016-03-02
14	Cao Pi	50	2016-09-11
15	Cao Pi	10	2016-09-12
```

<!-- more-->

1. 切换数据库

```sql
use test;
```

2. 显示数据库中的表

```sql
show tables;
```

3. 如果已经存在`user`表，删除它

```sql
drop table user;
```

4. 创建表

```sql
create table user(
    id int(11) auto_increment primary key,
    name varchar(200) not null,
    price numeric(20,2) default 0.0,
    update_time date
);
```
创建表的时候基本语法是： 列名 类型 [约束]，

约束是可选的，可用的约束有：

- unique  唯一值
- not null 不能为空
- default 默认值  添加默认值
- check 满足某一个条件
- primary key 指定为主键
- foreign key 指定外键

只约束某一列的时候，可以直接写在某个列上，如果涉及多个列，要写在最后。

```sql
create table user(
    id int(11) primary key,
    name varchar(200) not null,
    price numeric(20,2) default 0.0,
    update_time date,
    check (id > 0 and price > 0),
);
```

5. 插入值

```sql
insert into user (name, price , update_time) values ('Li Xiao',10.0,'2017-11-01');
```

指定自动增长的列和没有设置非空约束的列可以没有对应的值，会自动添加对应的数据。指定非空的列在插入的时候必须有值。

6. 更新值

```sql
update user set update_time = curdate() where name = 'Li Xiao';
```

7. 删除值

```sql
delete from user where name = 'Li Xiao';

```

完整的创建数据库和插入需要的数据的语句：

```sql
create table user(
  id int(11) primary key auto_increment,
  name varchar(100) not null,
  price numeric default 0.0,
  date date 
);
insert into user (name,price,date) values ('Li Xiao',10.2,curdate());
insert into user (name,price,date) values ('Li Xiao',80.0,curdate());
insert into user (name,price,date) values ('Li Xiao',10.0,curdate());
insert into user (name,price,date) values ('Zhao Hi',20.0,'2016-10-10');
insert into user (name,price,date) values ('Zhao Hi',20.0,'2016-6-10');
insert into user (name,price,date) values ('Zhao Hi',20.0,'2016-3-6');
insert into user (name,price,date) values ('Kao Ha',10.0,'2016-3-5');
insert into user (name,price,date) values ('Kao Ha',20.0,'2016-3-7');
insert into user (name,price,date) values ('Cao Pi',50.0,'2016-9-10');
insert into user (name,price,date) values ('Cao Pi',10.0,'2016-2-10');
```

8. 查询

每个人的总额

```sql
select name, sum(price) as total from user group by name;
```

每个人的记录数量

```sql
select name, count(*) as nums from user group by name;
```

每个人的平均值

```sql
select name, sum(price) / count(*) as avg from user group by name;

```

总的金额

```sql
select sum(price) from user;
```

每个人每次得到的钱占总共金额的百分比

```sql
select name, price / (select sum(price) from user) as percent from user;
```

每个人得到的钱的和占总金额的百分比：

```sql
select name, sum(price) / (select sum(price) from user) as percent from user group by name;
```

输出记录数量大于2且总金额大于90的人的姓名

```sql
select name from user group by name having count(*) > 2 and sum(price) > 90;
```

输出2016年每个人得到的金额 占 2016 年总金额的百分比

```sql
select name, sum(price) / (select sum(price) from user where extract(year from date) = 2016) as percent from user where extract(year from date) = 2016 group by name;
```

输出每一年 每个人得到的总金额 占当年总金额的百分比

```sql
1. 获得每个人在一年总的收入金额
(select name, sum(price) as price ,extract(year from date) as year from user group by name)；

2. 获得每一年总的金额
select extract(year from date) as year , sum(price) as sum from user group by extract(year from date);
3. 两个表连接

select a.name, a.price, b.sum, a.year from ((select name, sum(price) as price ,extract(year from date) as year from user group by name)) as a, (select extract(year from date) as year , sum(price) as sum from user group by extract(year from date)) as b where a.year = b.year;

4. 查询连接之后的表得到结果
select name, price / sum as percent from (select a.name, a.price, b.sum, a.year from ((select name, sum(price) as price ,extract(year from date) as year from user group by name)) as a, (select extract(year from date) as year , sum(price) as sum from user group by extract(year from date)) as b where a.year = b.year) as tmp

```