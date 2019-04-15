---
title: Spark环境下的Kmeans-Python实现
toc: true

tags:
  - spark
date: 2016-06-23 10:45:57
---
![spark](spark.png)
<!-- more -->
```scala
#设置应用名称，显示在spark监控页面上
sc = SparkContext(appName="MySparkApplication")

#读取数据,data文件夹下有6个数据文件，这样写能全部读取，需要注意的是，在其他worker的相同路径下也需要有这些文件
lines = sc.textFile("/home/fei/sparkcode/data/")

#原来的数据使用TAB分割的，现在把它转换成python中list形式
data = lines.map(lambda x:x.split('\t'))

#定义一个函数，将源文件中用字母数字表示的天数化成数字，转换后的格式为天数0-48，网站0-9。
def dayToNum(data):
    list1 = []
    list1.append(data[0])
    strs = data[1]
    day = (int(strs[1])-1)*7 + int(strs[3])-1
    list1.append(day)
    sitestr = data[2]
    site = int(sitestr[1:])-1
    list1.append(site)
    list1.append(int(data[3]))
    return list1

#使用定义的函数转换data中的数据
data2 = data.map(dayToNum)

#合并相同用户的数据，一个用户占一条记录
data4 = data3.reduceByKey(lambda v1,v2:list(map(lambda x: x[0]+x[1], zip(v2, v1))))
#保存合并后的数据，从这个数据能够知道有多少个用户，该用户7周对10个网站的点击情况是什么，保存下来便于以后需要直接处理
data4.saveAsTextFile("/home/fei/combinedData")
#为kmeans聚类准备数据
data5 = data4.map(lambda x:np.array(x[1]))

#设置kmeans参数：K=100，initializationMode="k-means||"
k = 100
mode = "k-means||"

#开始kmeans聚类
model = KMeans.train(data5,k,initializationMode=mode)
#保存聚类结果
output = open('/home/fei/kmeans_result.txt', 'w')
output.write("Final centers: " + str(model.clusterCenters))
output.write("Final centers: " + str(model.clusterCenters))
output.close()

#输出聚类结果
print("Final centers: " + str(model.clusterCenters))
print("Total Cost: " + str(model.computeCost(data5)))

#聚类完成后，预测每个用户的聚类类别ID：#reduceByKey会对相同key的记录进行reduce，这里将类别作为key,便于依据类别分别处理
belongs= data5.map(lambda x:(model.predict(x),(list)(x)))

#为线性回归准备训练数据
data6 = belongs.map(lambda x: (x[0],np.array(x[1]).reshape(8,7,10))).cache()
data00 = data6.map(lambda x:x[1]).map(lambda x:x[:,0,0])
data_train = data00.map(lambda x : LabeledPoint(x[6],x[:6]))

#开始训练
model = LinearRegressionWithSGD.train(data_train)

#预测并保存预测结果
result = model.predict(data[1][1:7,0,0])
result.saveAsTextFile('/home/fei/lines')

#停止程序
sc.stop()
```
