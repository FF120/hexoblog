# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 20:47:47 2017

@author: FF120
"""
import numpy as np
import math as mm

class LogisticRegression():
    """
    theta : 初始的参数选择(n+1_features),最后一个参数是截距b
    alpha : 学习率
    epsilon : 终止条件
    """
    def __init__(self,theta,alpha,epsilon):
        self.theta = theta
        self.alpha = alpha
        self.epsilon = epsilon
    
    def sigmoid(z):
        return 1.0 / (1 + mm.exp(-z))
    
    def update_theta(self,X,y):
        """
        使用所有训练数据完成一次参数的更新过程
        
        返回值
        epsilon: 本次更新的梯度和上次的差异
        """
        m,n = X.shape
        for j in range(n+1): # 第j个参数的偏移
            delta_theta = np.zeros((1,n+1))
            for i,line_x,line_y in enumerate(zip(X,y)):
                line_x = np.array(list(line_x) + [1]) # 添加x_n+1
                theta = np.array(self.theta)
                delta_theta[i] += self.sigmoid(np.dot(theta.T,line_x) - line_y) * (line_x[j])
                
            delta_theta = delta_theta*(1.0 / m)
            epsilon = np.sum(np.absolute(self.alpha *delta_theta))
            self.theta = self.theta + self.alpha * delta_theta
            return epsilon
            
    def fix(self,X,y):
        """
        X : (n_samples,n_featues) 训练集合
        y : (n_samples,) 训练集对应的标签
        """
        while True:
            epsilon = self.update_theta(X,y)
            if epsilon < self.epsilon:
                break
                return
        
    def predict(self,X):
        """
        X : (n_samples.n_features) 训练集合
        """
        if X.shape[0] == 1:
            y = self.sigmoid( np.dot(self.theta.T,X) )
            if y >= 0.5:
                return 1;
            if y < 0.5:
                return 0;