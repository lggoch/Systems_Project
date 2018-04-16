from implementations import *

import grpc
import sgd_pb2_grpc
import sgd_pb2
from random import shuffle, randint
from collections import Counter
import operator
import threading as th
import queue

# Reading of the files 
print('Reading the files')
file1 = open("../lyrl2004_vectors_test_pt0.dat")
file2 = open("../lyrl2004_vectors_test_pt1.dat")
file3 = open("../lyrl2004_vectors_test_pt2.dat")
file4 = open("../lyrl2004_vectors_test_pt3.dat")
files = [file1, file2, file3, file4]
# Parsing the files and storing the ids of the articles along with their respective features
id_ = []
samples = []
for f in files:
    for i in f.readlines():
        id_.append(i.split()[0])
        samples.append(i.strip().split()[2:])
categories = open("../rcv1-v2.topics.qrels")
print('Done!')

# Get a mapping of each article id to the set of categories it belongs to
print('Mapping into categories')
cat = []
cat_count = {}
id_to_labels = {}
for line in open("../rcv1-v2.topics.qrels").readlines():
    s = line.split(' ')
    id_to_labels.setdefault(s[1],[]).append(s[0])
    cat_count.setdefault(s[0],[]).append(s[1])
    cat.append(s[0])
print('Done!')

# Set the label vector based on the chosen category
# Set the features matrix as sparse matrix by storing only the non zero components in a dict 
print('Setting the labels')

y = set_labels("ECAT", id_, id_to_labels)
X = []
for sample in samples:
    d = dict()
    for feature in sample:
        key = int(feature.split(':')[0])
        value = float(feature.split(':')[1])
        d[key] = value
    X.append(d)

print('Spltting the data')
# Split the data into training and test sets
x_train, y_train, x_test, y_test = split_data(X, y, ratio=0.8)

# Compute the 'balancing' ratios (used to fairly penalize the negative vs positive examples)
c1, c2 = inbalance(y_train)

# Pick a random sample from the training set
rd = randint(0,len(x_train))    
batch = x_train[rd]
lab = y_train[rd]

# Set up a channel for the communication
print('Setting up a channel')
channel1 = grpc.insecure_channel('localhost:50051')
#channel2 = grpc.insecure_channel('localhost:52251')
stub1 = sgd_pb2_grpc.SGDStub(channel1)
#stub2 = sgd_pb2_grpc.SGDStub(channel2)

# Define the hyperparameters for the SGD

max_iter = 10000
worker = 10
q = queue.Queue()
weights = zeros(47237)
l_rate = 0.3
lambda_ = 0.001

print('Starting SGD...')
# Performs the SVM using SGD
for i in range(max_iter):
    results = []
    threads = []
    for j in range(worker):
        rd = randint(0,len(x_train))
        batch = x_train[rd]
        lab = y_train[rd]
        args_ = (sgd_pb2.LWB(labels=lab, weights=weights, batch=batch, corr_1=c1, corr_2=c2, lambda_= lambda_),q,j)
        t = th.Thread(target=test, args=args_)
        threads.append(t)
        t.start()
    for j in range(worker):
        threads[j].join()
    
    while not q.empty():
        results.append(q.get())
    for update in results:
        for x in update:
            weights[x] -= l_rate * update[x]
print('Done!')
print('*------------------------------------------------*')
print('Training accuracy = {}'.format(accuracy(prediction(x_train, weights), y_train)))
print('Test accuracy = {}'.format(accuracy(prediction(x_test, weights), y_test)))
print('*------------------------------------------------*')