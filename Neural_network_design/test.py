# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:08:41 2022

@author: TracyDong
"""

import neural_network
#导入neural_network.py,使用该文件中定义的类

input_nodes=3

hidden_nodes=3

output_nodes=3

learning_rate=0.3

#构建NeuralNetwork类
n = neural_network.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#输入用于测试的数据，输出经过模型得到的结果
print(n.testing([1.0, 0.5, -1.5]))