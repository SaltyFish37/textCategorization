import numpy as np
import pickle
import math

def ini_para(vector_size):
    w = np.zeros(vector_size)
    b = 0.0
    return w, b


def sign(x, w, b):
    result = 0.0
    for i in x:
        result += x[i] * w[i]
    result = result + b
    return result


def update_w(w, x, y, yita):
    for i in x:
        w[i] += yita * y * x[i]
    return w


def MLP_train(x, y, yita, vector_size):
    w, b = ini_para(vector_size)
    limit = math.floor(vector_size * 0.001) # 允许错误的存在
    is_wrong = False
    while not is_wrong:
        wrong_count = 0
        for i in range(len(x)):
            X = x[i]
            Y = y[i]
            if Y * sign(X,w,b) <= 0:
                w = update_w(w, X, Y, yita)
                b = b + yita * Y
                wrong_count += 1
        if wrong_count < limit:
            is_wrong = True
    return w, b


def label(l, y_positive):
    for i in range(len(l)):
        if l[i] != y_positive:
            l[i] = -1
        else:
            l[i] = 1
    return l

