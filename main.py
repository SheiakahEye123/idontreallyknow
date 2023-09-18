import math

import mnist
import numpy as np
import matplotlib.pyplot as plt
import random

train = mnist.download_and_parse_mnist_file('train-images.idx3-ubyte',
                                            r'C:\Users\kaide\PycharmProjects\testtestandmoretest\archive')
trainlabels = mnist.download_and_parse_mnist_file('train-labels.idx1-ubyte',
                                                  r'C:\Users\kaide\PycharmProjects\testtestandmoretest\archive')
test = mnist.download_and_parse_mnist_file('t10k-images.idx3-ubyte',
                                           r'C:\Users\kaide\PycharmProjects\testtestandmoretest\archive')
testlabels = mnist.download_and_parse_mnist_file('t10k-labels.idx1-ubyte',
                                                 r'C:\Users\kaide\PycharmProjects\testtestandmoretest\archive')

LAYER_COUNT = 2
LAYER_SCALE = 16
RELU_SCALE = 0.001

train = train.astype(float)
test = test.astype(float)

plt.rcParams["figure.figsize"] = [0, 1]


def leaky_relu(x):
    x[x <= 0] = x[x <= 0] / 1000
    return x


def sigma_oid(x): return 1 / (1 + np.power(math.e, -x))


def neuron_activation(A, W, B): return sigma_oid(np.sum(np.array(W) * np.array(A)) + B)


def layer(input, layernumber):
    if layernumber == LAYER_COUNT: return [neuron_activation(input, weights[layernumber], biases[layernumber][i])
                                               for i in range(10)]
    return [neuron_activation(input, weights[layernumber], biases[layernumber][i]) for i in range(LAYER_SCALE)]

def cost(input, correctlayer): return np.square(input - correctlayer)


indicee = random.randint(0, 59999)
weights = [[1 for i in range(LAYER_SCALE)] for e in range(LAYER_COUNT + 1)]
weights[0] = np.ones_like(train[0])


biases = [[0 for x in range(LAYER_SCALE)] for y in range(LAYER_COUNT)]
biases.append([0 for z in range(10)])



layers = [layer(train[indicee], 0)]
layers.append(layer(layers[0], 1))
layers.append(layer(layers[1], 2))

print(layers)

# for i in range(LAYER_COUNT): layers.append(layer(layers[i-1], i+1))
# print(weights)
# print(biases)
# layers.append(layer(layers[i-1], LAYER_COUNT))
# https://www.youtube.com/watch?v=IHZwWFHWa-w&list=TLPQMTcwOTIwMjP58GT4ZekNuQ&index=4
# [1 for i in range(LAYER_SCALE)] for e in range(LAYER_COUNT + 2)]