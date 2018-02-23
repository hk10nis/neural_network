# -*- coding: utf-8 -*
import numpy as np
from mnist import load_mnist


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def softmax(x):
    c  = np.max(x)
    y = np.exp(x-c) / np.sum( np.exp(x-c) )
    return y

def cross_entropy_error(x,t):
    batch_size = x.shape[0]
    y = -np.sum(t * np.log(x)) / batch_size
    return y

neuron_num = 50

b1 = np.zeros(neuron_num)
b2 = np.zeros(10)
w1 = 0.01 * np.random.randn(784,neuron_num)
w2 = 0.01 * np.random.randn(neuron_num,10)


(x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 100

#ノイズ生成
"""
import random
index = range(784)
for j in x_train:
    index_with_noise = random.sample(index,157)
    for i in index_with_noise:
        j[i] = (random.randint(0,255))/255.0
print("completed noise-adding")
"""
for i in range(20):
    for i in range(60000):

        x = x_train[i]
        t = t_train[i]
        #フィードフォワード
        layer_1_1 = np.dot(x , w1) + b1
        layer_1_2 = sigmoid(layer_1_1)
        layer_2 = np.dot(layer_1_2 , w2) + b2

        layer_3 = softmax(layer_2)
        y = cross_entropy_error(layer_3,t)


        #誤差逆伝播法
        #softmax,cross enrtopy error層の逆伝播
        reverse_layer_3 = layer_3 - t
        #
        db2 = reverse_layer_3
        dw2 = np.dot(layer_1_2.reshape(neuron_num,1), reverse_layer_3.reshape(1,10))
        reverse_layer_2_1 = np.dot(reverse_layer_3,w2.T)

        #sigmoidの逆伝播
        reverse_layer_1_2 = reverse_layer_2_1 * (1.0 - layer_1_2) * layer_1_2
        #
        db1 = reverse_layer_1_2
        dw1 = np.dot(x.reshape(784,1),reverse_layer_1_2.reshape(1,neuron_num))


        #パラメータ更新
        learning_rate = 0.1
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2

accuracy_counter = 0
print ("testing")
for i in range(10000):
    x = x_test[i]
    y = t_test[i]
    layer_1_1 = np.dot(x , w1) + b1
    layer_1_2 = sigmoid(layer_1_1)
    layer_2_1 = np.dot(layer_1_2 , w2) + b2
    layer_3 = softmax(layer_2_1)
    if np.argmax(layer_3) == np.argmax(y):
        accuracy_counter += 1

print(accuracy_counter/100.0,"%")
