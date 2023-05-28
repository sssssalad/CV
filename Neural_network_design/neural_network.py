# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import scipy.special as spl

class NeuralNetwork:
    """初始化模型结构"""
    
    def __init__(self, numInputNodes, numHiddenNodes, 
                 numOutputNodes, learningRate):
        
        """初始化模型输入层、隐藏层和输出层结点个数，
           初始化学习率、输入层到隐藏层的权重矩阵和隐藏层到输出层的权重矩阵，
           设置模型的激活函数是sigmoid（）"""
        
        self.numInputNodes  = numInputNodes        
        self.numHiddenNodes = numHiddenNodes        
        self.numOutputNodes = numOutputNodes       
        self.learningRate =learningRate
        
        #随机初始化权重矩阵
        #numpy.random.rand()生成取值范围为[0,1)的数组，-0.5让权重矩阵范围为[-0.5,0.5)
        #生成shape为（self.numHiddenNodes，self.numInputNodes）的权重矩阵
#         self.weightInputHidden = (numpy.random.rand(self.numHiddenNodes, 
#                                                     self.numInputNodes) -0.5)
#         self.weightHiddenOutput = (numpy.random.rand(self.numOutputNodes, 
#                                                      self.numHiddenNodes) -0.5)
        
        #正态初始化权重矩阵
        #均值为0，标准差为隐藏层神经元个数的平方根
        #size为（隐藏层神经元个数，输入层神经元个数）
        self.weightInputHidden = (numpy.random.normal(0.0, pow(self.numHiddenNodes,
                                                               -0.5), 
                                                      (self.numHiddenNodes,
                                                       self.numInputNodes)))
        self.weightHiddenOutput = (numpy.random.normal(0.0, pow(self.numOutputNodes,
                                                                -0.5),
                                                       (self.numOutputNodes,
                                                        self.numHiddenNodes)))
        #设置激活函数sigmoid（）
        #expit 函数，逻辑 sigmoid 函数，定义为 expit(x) = 1/(1+exp(-x)) 。
        
        #匿名函数
        self.activation_function = lambda x: spl.expit(x)
        
#         #等价于
#         def activation_function(self, x):
#             return scipy.special.expit(x)
    
#         def sigmoid(x):
#         """
#         这个函数的作用就是把数据压缩到0至1之间，成S型分布的曲线
#         param：x
#         return: 0-1之间的小数，不包括0和1
#         """
#         return scipy.special.expit(x)
        
    def training(self, inputs_list, targets_list):
        """输入训练集数据x_train和其真实值y_train，
           分别计算隐藏层和输出层的输入、输出，
           计算模型误差、隐藏层误差和每一层的误差，
           并将每一层的误差应用于权重矩阵的参数更新"""
        
        #ndim为维度的个数，输入矩阵是二维数组
        inputs= numpy.array(inputs_list, ndmin=2).T
        targets= numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs =numpy.dot(self.weightInputHidden, inputs)        
        hidden_outputs= self.activation_function(hidden_inputs)   
        
        final_inputs =numpy.dot(self.weightHiddenOutput, hidden_outputs)       
        final_outputs= self.activation_function(final_inputs)
        
        #模型误差，真实值-模型预测值
        output_errors= targets - final_outputs
        
        #隐藏层误差，隐藏层到输出层的权重矩阵*模型误差
        hidden_errors=numpy.dot(self.weightHiddenOutput.T, output_errors)
        
        #输出权重矩阵参数更新
        #每一层的误差：（真实值-模型预测值）*模型预测值*（1-模型预测值）
        #np.transpose()作用:改变序列,这里是将矩阵转置
        self.weightHiddenOutput += self.learningRate * numpy.dot((output_errors * 
                                                                  final_outputs *
                                                                  (1.0 - final_outputs)),
                                                                 numpy.transpose(hidden_outputs))
        
        #输入权重矩阵参数更新
        self.weightInputHidden += self.learningRate * numpy.dot((hidden_errors * 
                                                                 hidden_outputs *
                                                                 (1.0 - hidden_outputs)),
                                                                numpy.transpose(inputs))

    def testing(self, inputs_list):
        """输入测试集数据x_test和y_test,
           分别计算隐藏层和输出层的输入、输出"""
        
        inputs= numpy.array(inputs_list, ndmin=2).T
        
        hidden_inputs =numpy.dot(self.weightInputHidden, inputs)
        
        hidden_outputs= self.activation_function(hidden_inputs)
        
        final_inputs =numpy.dot(self.weightHiddenOutput, hidden_outputs)
        
        final_outputs= self.activation_function(final_inputs)
        
        return final_outputs
