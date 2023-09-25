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


def neuron_activation(A, W, B):    return sigma_oid(np.sum(np.array(W) * np.array(A)) + B), np.sum(
    np.array(W) * np.array(A)) + B


def layer(input, layernumber):
    if layernumber == LAYER_COUNT: return [neuron_activation(input, weights[layernumber], biases[layernumber][i])
                                           for i in range(10)]
    return [neuron_activation(input, weights[layernumber], biases[layernumber][i]) for i in range(LAYER_SCALE)]


def cost(input, correctlayer): return np.sum(np.square(np.array(input).T[0] - np.array(correctlayer)))


# def backprop():


weights = [[[0, 0] for i in range(LAYER_SCALE)] for e in range(LAYER_COUNT + 1)]
weights[0] = np.ones_like(train[0])

biases = [[0 for x in range(LAYER_SCALE)] for y in range(LAYER_COUNT)]
biases.append([0 for z in range(10)])


def trainone(index):
    layers = [layer(train[index], 0)]
    for i in range(LAYER_COUNT): layers.append(layer(layers[i], i + 1))
    correctlayer = np.zeros(10)
    correctlayer[trainlabels[index] - 1] = 1.00
    coast = cost(layers[LAYER_COUNT], correctlayer)
    print(coast)
    plt.figure(0)
    plt.imshow(train[index])
    plt.show()

#backprop code:
def backprop(cost, layers, weights):

    slope = layer[:][1]


trainone(3)
