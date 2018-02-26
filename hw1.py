#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def soft_max(x):
    exp_s = np.exp(x)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# todo: Problem_1
def load_data(path):
    txt = np.loadtxt(path, delimiter=',')
    Y = txt[:, 784].astype(int)  # (1,3000)
    X = np.delete(txt, -1, axis=1)  # (3000,784)
    return X, Y


# todo: Problem_2
def update_weights_perceptron(X, Y, weights, bias, lr):
    classes = 10
    # change weights bias to array
    W = np.array(weights)[0]  # 784,10
    B = np.array(bias)[0]  # 1,10

    y_hat = np.eye(classes)[Y]  # Y -> (3000,10)
    # softmax compute (forward prop)
    z = np.dot(X, W) + B  # (3000,10)
    output = soft_max(z)  # (3000,10)  y = softmax(wx + b) activation
    dgradient = (output - y_hat) / Y.shape[0]  # 3000  # Y(3000,10)

    dW = np.dot(X.T, dgradient)  # (784,10)
    dB = np.sum(dgradient, axis=0)  # (1,10) take the median number as dB

    updated_weights = [W - dW * lr]
    updated_bias = [B - dB * lr]

    return updated_weights, updated_bias


# todo: Problem_3
def update_weights_single_layer(X, Y, weights, bias, lr):
    # INSERT YOUR CODE HERE
    classes = 10
    y_hat = np.eye(classes)[Y]  # Y -> (3000,10)
    w1 = weights[0]  # (784,10)
    w2 = weights[1]  # (10,10)
    b1 = bias[0]  # (1,10)
    b2 = bias[1]  # (1,10)

    z1 = np.dot(X, w1) + b1  # 3000,10
    sig = sigmoid(z1)  # 3000,10

    z2 = np.dot(sig, w2) + b2  # 3000,10
    output = soft_max(z2)  # 3000,10
    dg = (output - y_hat) / Y.shape[0]  # (3000, 10)

    print(z1.shape, sig.shape, z2.shape, y_hat.shape)
    print((1 - sig).shape)

    z3 = np.dot(dg, w2.T)
    z4 = sig * (1 - sig)
    hidden_layer = z3 * z4

    dw1 = np.dot(X.T, hidden_layer)
    db1 = np.sum(hidden_layer, axis=0)

    dw2 = np.dot(sig.T, dg)
    db2 = np.sum(dg, axis=0)

    updated_weights = [w1 - lr * dw1, w2 - lr * dw2]
    updated_bias = [b1 - lr * db1, b2 - lr * db2]

    return updated_weights, updated_bias


# todo: Problem 4
def update_weights_double_layer(X, Y, weights, bias, lr):
    # INSERT YOUR CODE HERE
    # print(X.shape)
    # print(Y.shape)
    # (784, 10) (10, 10) (10, 10)
    # print(weights[0].shape, weights[1].shape, weights[2].shape)
    # (1, 10) (1, 10) (1, 10)
    # print(bias[0].shape, bias[1].shape, bias[2].shape)
    classes = 10
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    b1 = bias[0]
    b2 = bias[1]
    b3 = bias[2]
    y_hat = np.eye(classes)[Y]

    z1 = np.dot(X, w1) + b1  # 3000,10
    sig1 = sigmoid(z1)  # 3000,10

    z2 = np.dot(sig1, w2) + b2  # 3000,10
    sig2 = sigmoid(z2)  # 3000, 10

    z3 = np.dot(sig2, w3) + b3  # 3000,10
    output = soft_max(z3)  # 3000, 10
    dg = (output - y_hat) / Y.shape[0]

    dz1 = sigmoid(z2) * (1 - sigmoid(z2))
    dz2 = np.dot(dg, w3.T) * dz1
    dz3 = sigmoid(z1) * (1 - sigmoid(z1))
    dz4 = np.dot(dz2, w2.T) * dz3

    dw1 = np.dot(X.T, dz4)
    db1 = np.sum(dz4, axis=0)

    dw2 = np.dot(sig1.T, dz2)
    db2 = np.sum(dz2, axis=0)

    dw3 = np.dot(sig2.T, dg)
    db3 = np.sum(dg, axis=0)

    updated_weights = [w1 - lr * dw1,
                       w2 - lr * dw2,
                       w3 - lr * dw3]
    updated_bias = [b1 - lr * db1,
                    b2 - lr * db2,
                    b3 - lr * db3]

    return updated_weights, updated_bias


# todo: Problem5
def update_weights_double_layer_act(X, Y, weights, bias, lr, activation):  # INSERT YOUR CODE HERE
    classes = 10
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    b1 = bias[0]
    b2 = bias[1]
    b3 = bias[2]
    y_hat = np.eye(classes)[Y]

    if activation == 'sigmoid':
        # INSERT YOUR CODE HERE
        z1 = np.dot(X, w1) + b1  # 3000,10
        sig1 = sigmoid(z1)  # 3000,10

        z2 = np.dot(sig1, w2) + b2  # 3000,10
        sig2 = sigmoid(z2)  # 3000, 10

        z3 = np.dot(sig2, w3) + b3  # 3000,10
        output = soft_max(z3)  # 3000, 10

        dg = (output - y_hat) / Y.shape[0]
        dz1 = sigmoid(z2) * (1 - sigmoid(z2))
        dz2 = np.dot(dg, w3.T) * dz1
        dz3 = sigmoid(z1) * (1 - sigmoid(z1))
        dz4 = np.dot(dz2, w2.T) * dz3

        dw1 = np.dot(X.T, dz4)
        db1 = np.sum(dz4, axis=0)

        dw2 = np.dot(sig1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dw3 = np.dot(sig2.T, dg)
        db3 = np.sum(dg, axis=0)

        updated_weights = [w1 - lr * dw1,
                           w2 - lr * dw2,
                           w3 - lr * dw3]
        updated_bias = [b1 - lr * db1,
                        b2 - lr * db2,
                        b3 - lr * db3]

    if activation == 'tanh':
        # INSERT YOUR CODE HERE
        z1 = np.dot(X, w1) + b1  # 3000,10
        sig1 = np.tanh(z1)  # 3000,10

        z2 = np.dot(sig1, w2) + b2  # 3000,10
        sig2 = np.tanh(z2)  # 3000, 10

        z3 = np.dot(sig2, w3) + b3  # 3000,10
        output = soft_max(z3)  # 3000, 10

        dg = (output - y_hat) / Y.shape[0]
        dz1 = 1 - np.tanh(z2) * np.tanh(z2)
        dz2 = np.dot(dg, w3.T) * dz1
        dz3 = 1 - np.tanh(z1) * np.tanh(z1)
        dz4 = np.dot(dz2, w2.T) * dz3

        dw1 = np.dot(X.T, dz4)
        db1 = np.sum(dz4, axis=0)

        dw2 = np.dot(sig1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dw3 = np.dot(sig2.T, dg)
        db3 = np.sum(dg, axis=0)

        updated_weights = [w1 - lr * dw1,
                           w2 - lr * dw2,
                           w3 - lr * dw3]
        updated_bias = [b1 - lr * db1,
                        b2 - lr * db2,
                        b3 - lr * db3]

    if activation == 'relu':
        # INSERT YOUR CODE HERE
        def relu(x):
            return np.where(x > 0, x, 0.0)

        def drelu(x):
            return np.where(x > 0, 1.0, 0.0)

        z1 = np.dot(X, w1) + b1  # 3000,10
        sig1 = relu(z1)  # 3000,10

        z2 = np.dot(sig1, w2) + b2  # 3000,10
        sig2 = relu(z2)  # 3000, 10

        z3 = np.dot(sig2, w3) + b3  # 3000,10
        output = soft_max(z3)  # 3000, 10

        dg = (output - y_hat) / Y.shape[0]
        dz1 = drelu(z2)
        dz2 = np.dot(dg, w3.T) * dz1
        dz3 = drelu(z1)
        dz4 = np.dot(dz2, w2.T) * dz3

        dw1 = np.dot(X.T, dz4)
        db1 = np.sum(dz4, axis=0)

        dw2 = np.dot(sig1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dw3 = np.dot(sig2.T, dg)
        db3 = np.sum(dg, axis=0)

        updated_weights = [w1 - lr * dw1,
                           w2 - lr * dw2,
                           w3 - lr * dw3]
        updated_bias = [b1 - lr * db1,
                        b2 - lr * db2,
                        b3 - lr * db3]

    return updated_weights, updated_bias


# todo: Problem6
def update_weights_double_layer_act_mom(X, Y, weights, bias, lr, activation, momentum, epochs):  # INSERT YOUR CODE HERE
    classes = 10
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    b1 = bias[0]
    b2 = bias[1]
    b3 = bias[2]
    y_hat = np.eye(classes)[Y]
    vw = [0, 0, 0]
    vb = [0, 0, 0]

    if activation == 'sigmoid':
        # INSERT YOUR CODE HERE
        activate_f = lambda x: sigmoid(x)
        back_activate_f = lambda x: sigmoid(x) * (1 - sigmoid(x))

    if activation == 'tanh':
        # INSERT YOUR CODE HERE
        activate_f = lambda x: np.tanh(x)
        back_activate_f = lambda x: 1 - np.tanh(x) * np.tanh(x)

    if activation == 'relu':
        # INSERT YOUR CODE HERE
        activate_f = lambda x: (np.where(x > 0, x, 0.0))
        back_activate_f = lambda x: (np.where(x > 0, 1.0, 0.0))


    for i in range(epochs):
        z1 = np.dot(X, w1) + b1  # 3000,10
        sig1 = activate_f(z1)  # 3000,10

        z2 = np.dot(sig1, w2) + b2  # 3000,10
        sig2 = activate_f(z2)  # 3000, 10

        z3 = np.dot(sig2, w3) + b3  # 3000,10
        output = soft_max(z3)  # 3000, 10

        dg = (output - y_hat) / Y.shape[0]
        dz1 = back_activate_f(z2)
        dz2 = np.dot(dg, w3.T) * dz1
        dz3 = back_activate_f(z1)
        dz4 = np.dot(dz2, w2.T) * dz3

        dw1 = np.dot(X.T, dz4)
        db1 = np.sum(dz4, axis=0)

        dw2 = np.dot(sig1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dw3 = np.dot(sig2.T, dg)
        db3 = np.sum(dg, axis=0)

        vw[2] = momentum * vw[2] - dw3 * lr
        vb[2] = momentum * vb[2] - db3 * lr

        vw[1] = momentum * vw[1] - dw2 * lr
        vb[1] = momentum * vb[1] - db2 * lr
        
        vw[0] = momentum * vw[0] - dw1 * lr
        vb[0] = momentum * vb[0] - db1 * lr

        if i == 0:
            update_w1 = w1 + vw[0]
            update_w2 = w2 + vw[1]
            update_w3 = w3 + vw[2]
            update_b1 = b1 + vb[0]
            update_b2 = b2 + vb[1]
            update_b3 = b3 + vb[2]
        else:
            update_w1 += vw[0]
            update_w2 += vw[1]
            update_w3 += vw[2]
            update_b1 += vb[0]
            update_b2 += vb[1]
            update_b3 += vb[2]

        updated_weights = [update_w1,
                           update_w2,
                           update_w3]
        updated_bias = [update_b1,
                        update_b2,
                        update_b3]

    return updated_weights, updated_bias
