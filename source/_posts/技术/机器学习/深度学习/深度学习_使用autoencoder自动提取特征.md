---
title: 深度学习_使用autoencoder自动提取特征
toc: true

tags:
  - autoencoder
  - fmri
date: 2017-04-27 19:19:11
---
autoencoder自动提取特征的原理是使用输入数据作为输出标签训练数据，中间层的神经元的数目小于特征数目，使用中间层的神经元的输出作为特征，这样就达成一个特征空间变换和降维的目的。本质上它是通过一种非线性的变换函数在转换特征空间。
<!-- more -->

## 实现一个最简单的自动特征提取器
```python
# 准备数据
X_root = r'd:/X.npy'
X = np.load(X_root)
root = r'D:\FMRI_ROOT\YIYU\CONN\conn_project04\meresult'#####################
y_path = r'y.npy'

# 加载数据转成python格式
data_root = os.path.join(root,'data')
y = np.load(os.path.join(data_root,y_path))


from sklearn import cross_validation
test_size = 0.3
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=test_size, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
###############################################################################
###############################################################################
# 定义神经网络的结构
encoding_dim = 10

# this is our input placeholder
input_img = Input(shape=(4298,))

encoded = Dense(1000, activation='relu')(input_img)
encoded = Dense(100, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(100, activation='relu')(encoded)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(4298, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

save_model(autoencoder,'autoencoder')
save_model(encoder,'encoder')
save_model(decoder,'decoder')
## 加载保存的模型
autoencoder = load_model("autoencoder")
encoder = load_model("encoder")
decoder = load_model("decoder")

## 使用训练好的模型
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

## 使用分类器分类
from sklearn.svm import SVC
svc_linear =  SVC(C=1,kernel="linear")

from sklearn.cross_validation import KFold
cv = KFold(y_test.shape[0], n_folds=10)
score_linear_svc = []
for train,test in cv:
        # ----------------------------------------------------------
        svc_linear.fit(encoded_imgs[train],y_test[train])
        score_linear_svc.append( svc_linear.score(encoded_imgs[test],y_test[test]) )

print(np.mean(score_linear_svc))
```
