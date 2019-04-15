---
title: 深度学习_Keras使用技巧
toc: true

tags:
  - keras
date: 2017-04-19 22:06:50
---
keras是一个构建在更高水平上的深度学习框架，类似scikit-learn,他提供了高级的抽象的接口，使得你在不了解具体细节的基础上，就可以定义和使用各种神经网络算法完成某个具体的工作。
<!--more-->
## 模型的可视化
http://blog.csdn.net/u014749291/article/details/54891087
```python
Traceback (most recent call last):

  File "<ipython-input-7-779239671584>", line 1, in <module>
    SVG(model_to_dot(model).create(prog='dot', format='svg'))

  File "C:\ProgramData\Anaconda3\envs\python35\lib\site-packages\keras-2.0.3-py3.5.egg\keras\utils\vis_utils.py", line 35, in model_to_dot
    _check_pydot()

  File "C:\ProgramData\Anaconda3\envs\python35\lib\site-packages\keras-2.0.3-py3.5.egg\keras\utils\vis_utils.py", line 17, in _check_pydot
    raise ImportError('Failed to import pydot. You must install pydot'

ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
```

如果是在Windows平台测试，很容易出现上述问题，下载相应的[软件](http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi)安装，再设置环境变量就可以了。


- 显示模型的信息： `model.summary()`;
- 获得模型训练好的参数信息: `model.get_weights()`
- 保存模型的图片到本地： `from keras.utils import plot_model plot_model(model, to_file='model.png')`
- `model.layers` is a flattened list of the layers comprising the model graph.
- `model.inputs` is the list of input tensors.
- `model.outputs` is the list of output tensors.

## 利用TensorBoard 实现模型的和计算过程的可视化
如果keras使用的是tensorflow作为后端的框架，我们可以借助TensorBoard实现高级一些的可视化。keras实现的可视化功能有限，不能满足深入研究和调试代码的要求。
[官方教程](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

1. 打开TensorBoard :

打开`cmd`,执行`activate python35`激活安装了tensorflow的环境，然后执行`tensorboard --logdir=path/to/logs`,在本地浏览器打开`http://localhost:6006/`可以看到下面的界面：
![](2017-04-21_103727.png)
## 模型持久化
模型的存储和模型的加载在编程中经常会用到，毕竟训练出一个模型常常需要很长的时间。模型的关键信息其实就两个方面，结构和参数。结构可以用一些结构化的字符串来表示，参数直接用numpy的结构就可以。keras分别提供了保存结构和保存参数的方法。

获得模型结构的方法：
```python
# 保存成json格式字符串
from models import model_from_json
json_string = model.to_json()

# 保存成 yaml格式的字符串
from models import model_from_yaml
yaml_string = model.to_yaml()

# 使用变量的形式返回结构的信息，可以自己使用其他工具序列化为字符串。
model.get_config()
```

获得模型的参数的方法：
```python
# 只保存模型的参数，不保存模型的结构
model.save_weights(filepath)
```

从已经保存的结构恢复模型的方法：
```python
from models import model_from_json
model = model_from_json(json_string)

from models import model_from_yaml
model = model_from_yaml(yaml_string)

# 从配置信息恢复模型
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```
为模型设置参数的方法：
```python
# 这里的model要和加载的权重保存的时候的模型结构一致。
model.load_weights(filepath, by_name=False)
```
我习惯使用的方法是：
```python
from keras.models import Model
# 保存模型
def save_model(model,name):
    ## fit完成之后，保存整个模型的配置
    root_path = r'd:\deeplearning'
    model_config = model.get_config()
    model_weight = model.get_weights()
    import pickle
    import os
    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    pickle.dump(model_config, open(config_save_path, 'wb'))
    pickle.dump(model_weight, open(weight_save_path, 'wb'))

# 加载模型
def load_model(name):
    root_path = r'd:\deeplearning'
    import pickle
    import os
    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    model_config = pickle.load(open(config_save_path, 'rb'))
    model_weight = pickle.load(open(weight_save_path, 'rb'))
    model = Model.from_config(model_config)
    model.set_weights(model_weight)
    return model
```
