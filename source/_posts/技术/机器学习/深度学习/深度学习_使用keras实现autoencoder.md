---
title: 深度学习_使用keras实现autoencoder
toc: true

tags:
  - keras
  - autoencoder
date: 2017-04-20 16:31:16
---
autoencoder是利用神经网络来自动提取有意义的特征的一种方法，它的基本思想是这样的： 输入和输出层拥有一样的神经元的数目，都等于数据的输入维数。中间拥有若干个隐含层，隐含层的神经元的数目要小于输入维数。使用输入作为输出训练整个神经网络，最后会得到一个$f(x)=x$的等式。观察训练好的神经网络，最后的输出是经过比输入的数据的维数更少的数据得到的，所以，中间隐含层的数据表示就是原来输入数据的某种抽象，因为压缩了数据的维数，所以是一种降维的方法。当然，也可以使输入输出的维数远远小于中间隐含层的维数，这样会得到低维数据的高维表示。
<!-- more -->
参考资料：https://blog.keras.io/building-autoencoders-in-keras.html

keras是一个抽象层次较高的深度学习框架，它以theano和tensorflow作为后端实现，使用她可以很方便的实现多种不同的autoencoder.
## 最简单的单层编码解码机
我们使用手写数字识别的数据构造一个简单的自动编码机。目的是给数据降维。我们构造一个三层的神经网络，输入层包含`28*28 = 784` 个神经元，中间的隐含层包含`4*4 = 16` 个神经元,输出层包含`28*28 = 784` 个神经元。然后把手写数字识别的数据去掉标签，后者说是把标签设置为自己的输入，然后训练该神经网络，就会得到一个把`28*28`的灰度图像映射到`4*4`的灰度图像的映射，通过后面两层神经网络，我们可以根据`4*4`的数据恢复原来的`28*28`的图像，如果恢复的图像的质量还不错，就能够说明神经网络自动学习到的压缩方法确实是很不错的。

```python
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pickle
import os
from keras import regularizers

def save_model(model,name):
    ## fit完成之后，保存整个模型的配置
    root_path = r'd:\deeplearning'
    model_config = model.get_config()
    model_weight = model.get_weights()

    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    pickle.dump(model_config, open(config_save_path, 'wb'))
    pickle.dump(model_weight, open(weight_save_path, 'wb'))

def load_model(name):
    root_path = r'd:\deeplearning'
    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    model_config = pickle.load(open(config_save_path, 'rb'))
    model_weight = pickle.load(open(weight_save_path, 'rb'))
    model = Model.from_config(model_config)
    model.set_weights(model_weight)
    return model


## 加载数据
(x_train, _), (x_test, _) = mnist.load_data()
# 把数据缩放到 0 -- 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

###############################################################################
###############################################################################
# this is the size of our encoded representations
# 将784维的数据压缩到32维
encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
# 新建了一个32个网络节点，激活函数使用relu的层，注意在此层前面应该是输入层，拥有784个神经元的节点。
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# 在压缩层添加稀疏的限制。在没有任何限制，只用节点的数目限制的时候，结果和PCA的结果十分类似
# 但是当加入稀疏优化之后，就不一样了
encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
# "decoded" is the lossy reconstruction of the input
# 在encoded的后面，新建了一个拥有784个节点的神经网络层，作为解码层。
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# 现在autoencoder是这样的一个神经网络，输入包含784个节点，紧接着是32个节点的第一个隐含层，然后是784个节点的输出层。

encoder = Model(input_img, encoded)
# encoder 以784个节点作为输入层，以32个节点作为输出层，没有中间的隐含层，这就是一个最简单的感知机的模型。

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
# 找到上面创建的神经网络的最后一层，也就是输出层。
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# 现在decoder 是以32个维度输入，以784个维度输出的简单的神经网络，需要注意的是，encoder和decorder都是上面定义的autoencoder神经网络的
# 的一部分，只是用了新的变量名称把其中的一部分网络提取出来了。所以，对应的网络参数是一样的，因为本来指代的就是同一个网络的不同部分。

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

##x训练过程，注意到这里并不是没有标签，而是把子集的输入作为了输出，所以这不是无监督学习。而是有监督学习
## 这里通常叫做自监督学习。
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

save_model(autoencoder,'autoencoder')
save_model(encoder,'encoder')
save_model(decoder,'decoder')
## 加载保存的模型
autoencoder = load_model("autoencoder")
encoder = load_model("encoder")
decoder = load_model("decoder")
## 这样，保存的模型包括结构和参数就都回来了。
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test) # 会利用上面训练好的网络，输出32维的数据
decoded_imgs = decoder.predict(encoded_imgs) # 会利用训练好的网络，输出784维的数据。

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
# 设置显示前多少个数据
n = 12  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

```

## 多层的编码解码机
单层神经网络的处理能力是有限的，无论编码还是解码，我们都可以使用多个层来提高映射，或者说数据表征空间转换的能力。下面我们建立一个多层的神经网络，让每层的神经元的数量逐渐减少再逐渐增大，这样可以实现一个分层的数据压缩，数据降维。 我们建立的神经网络的形式是这样的(`784-->128-->64-->32-->64-->128-->784`),包含输入层，输出层在内，一共有7层。
```python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:30:40 2017

@author: FF120
"""
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pickle
import os
from keras import regularizers

def save_model(model,name):
    ## fit完成之后，保存整个模型的配置
    root_path = r'd:\deeplearning'
    model_config = model.get_config()
    model_weight = model.get_weights()

    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    pickle.dump(model_config, open(config_save_path, 'wb'))
    pickle.dump(model_weight, open(weight_save_path, 'wb'))

def load_model(name):
    root_path = r'd:\deeplearning'
    config_save_path = os.path.join(root_path,name+"_config.txt")
    weight_save_path = os.path.join(root_path,name+"_weights.txt")
    model_config = pickle.load(open(config_save_path, 'rb'))
    model_weight = pickle.load(open(weight_save_path, 'rb'))
    model = Model.from_config(model_config)
    model.set_weights(model_weight)
    return model


## 加载数据
(x_train, _), (x_test, _) = mnist.load_data()
# 把数据缩放到 0 -- 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

###############################################################################
###############################################################################
encoding_dim = 32

# this is our input placeholder
input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(32,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

save_model(autoencoder,'autoencoder')
save_model(encoder,'encoder')
save_model(decoder,'decoder')
## 加载保存的模型
autoencoder = load_model("autoencoder")
encoder = load_model("encoder")
decoder = load_model("decoder")
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

上面的代码以中间神经元数目最少的作为分割，前面的作为编码机，后面的作为解码机，前后正好是对称的。当然，这并不是必须的，你也可以让中间的任何一层作为分割，前面作为编码机，后面的作为解码机。这样会实现不同的压缩表示层级。

我们把中间层向前推进一层，再看结果：
```python
input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(64,))
decoder_layer0 = autoencoder.layers[-4](encoded_input)
decoder_layer1 = autoencoder.layers[-3](decoder_layer0)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)
```

## 卷积神经网络的autoencoder
处理图像的时候，使用卷积神经网络通常可以有更好的效果。所以，我们尝试使用卷积神经网络做autoencoder,看看数据压缩的效果如何。

```python
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

## 28*28的图像，只有1个通道，黑白图像
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

# 定义一个卷积核是3*3的层
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# 定义一个pooling层，取2*2区域内的最大值
x = MaxPooling2D((2, 2), padding='same')(x)
# 定义一个3*3的核
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# 定义一个pooling层，取2*2区域内的最大值
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='d:/log')])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png')
```
上面的代码建立了一个这样的模型：
`input-->Conv2D-->MaxPooling2D-->Conv2D-->MaxPooling2D-->Conv2D-->MaxPooling2D(encoded)-->Conv2D-->UpSampling2D-->Conv2D-->UpSampling2D-->Conv2D-->UpSampling2D-->Conv2D(decoded)`
一共有14层网络。使用下面的语句可以打印出这些结构：
```python
from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png')
```
1[](model.png)
