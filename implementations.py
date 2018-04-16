import grpc
import sgd_pb2_grpc
import sgd_pb2
from random import shuffle, randint
from collections import Counter
import operator
import threading as th
import queue

def zeros(x):
    '''
    Returns a vector (in list type) consisting of x zeros
    '''
    return [0]*x

def hinge_loss(y, X, w):
    '''
    Computes the Hinge loss given:
    y: label vector
    X: feature vector
    w: weight vector
    '''
    loss = zeros(len(y))
    f = multiply_matrix(X, w)
    for i in range(len(y)):
        loss[i] = max(1 - y[i] * f[i], 0)
    return loss

def multiply(x, w):
    '''
    Perfoms the dot product between the vector x and w
    '''
    y_n = 0
    for k in x:
        y_n += x.get(k) * w[k]
    return y_n

def multiply_matrix(X, w):
    '''
    Perfoms the matrix multiplication X@w
    '''
    y = zeros(len(X))
    for i in range(len(X)):
        y[i] = multiply(X[i], w)
    return y

def prediction(X, w):
    '''
    Return the prediction labels (± 1) given:
    X: feature matrix
    w: weight vector
    '''
    y = multiply_matrix(X, w)
    y = [a > 0 for a in y]
    return [(a*2 - 1) for a in y]

def accuracy(y_pred, y):
    '''
    Computes the accuracy given:
    y_pred: predicted labels
    y: test labels
    '''
    return sum([i == j for (i, j) in zip(y_pred, y)])/len(y)

def calculate_primal(y, X, w, lambda_):
    '''
    Computes the primal loss with the regularizer term lambda
    '''
    v = hinge_loss(y, X, w)
    return sum(v) + lambda_ / 2 * sum([w_i**2 for w_i in w])

def set_labels(cat, id_, id_to_labels):
    '''
    Converts the article ids to a label vector for SVM given a category 'cat'
    '''
    labels = [1 if cat in id_to_labels[x] else -1 for x in id_]
    return labels

def split_data(tx, ty, ratio, seed=1):
    '''
    Splits the training data by ratio (dedicated to training)
    '''
    split_idxs = [i for i in range(len(tx))]
    
    # Shuffle the indices randomly
    shuffle(split_idxs)
    
    tx_shuffled = []
    ty_shuffled = []
    for i in range(len(split_idxs)):
        tx_shuffled.append(tx[split_idxs[i]])
        ty_shuffled.append(ty[split_idxs[i]])
    
    # Split by ratio
    split_pos = int(len(tx) * ratio)
    x_train = tx_shuffled[:split_pos]
    x_test = tx_shuffled[split_pos:]
    y_train = ty_shuffled[:split_pos]
    y_test = ty_shuffled[split_pos:]
    
    return x_train, y_train, x_test, y_test

def inbalance(labels):
    '''
    Computes the proportions of the 1s and (-1)s to balance penalizer terms
    when calculating the loss
    '''
    size = len(labels)
    c = Counter(labels)
    corr_1 = (0.5*size)/c[-1]
    corr_2= (0.5*size)/c[1]
    return (corr_1, corr_2)

def test(input_, q,j):
    res = stub1.ComputeTask(input_)
    q.put(dict(res.grad_up))