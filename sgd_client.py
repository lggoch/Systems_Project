from __future__ import print_function

import grpc

import sgd_pb2_grpc
import sgd_pb2
import numpy as np


#

def zeros(x):
	return [0]*x

def hinge_loss(y, X, w):
	loss = zeros(len(y))
	f = multiply_matrix(X, w)
	## Penalize but indicative for the prints
	for i in range(len(y)):
		loss[i] = max(1 - y[i] * f[i], 0)
	return loss

def accuracy(a, b):
	return float(sum([a_i == b_i for (a_i, b_i) in zip(a, b)])/len(a))

def calculate_primal(y, X, w, lambda_):
    v = hinge_loss(y, X, w)
    return np.sum(v) + lambda_ / 2 * sum([w_i**2 for w_i in w])

def set_labels(cat,id_, id_to_labels):
    #label_one = id_to_labels[cat]
    labels = [1 if cat in id_to_labels[x] else -1 for x in id_]
    return labels

def inbalance(labels):
    size = len(labels)
    c = Counter(labels)
    corr_1 = (0.5*size)/c[-1]
    corr_2= (0.5*size)/c[1]
    return (corr_1, corr_2)

def run():


	file1 = open("../lyrl2004_vectors_test_pt0.dat")
	file2 = open("../lyrl2004_vectors_test_pt1.dat")
	file3 = open("../lyrl2004_vectors_test_pt2.dat")
	file4 = open("../lyrl2004_vectors_test_pt3.dat")
	files = [file1,file2,file3,file4]
	id_ = []
	samples = []
	for f in files:
		for i in f.readlines():
			id_.append(i.split()[0])
			samples.append(i.strip().split()[2:])
	categories = open("../rcv1-v2.topics.qrels")
	id_to_labels = {}
	for line in categories.readlines():
		s = line.split(' ')
		if not s[1] in id_to_labels:
			id_to_labels.setdefault(s[0],[]).append(s[1])
	labels = set_labels("ECAT",id_, id_to_labels)
	c1,c2 = inbalance(labels)
	X = []
	for sample in samples:
		d = dict()
		for feature in sample:
			key = int(feature.split(':')[0])
			value = float(feature.split(':')[1])
			d[key] = value
		X.append(d)

	weights = zeros(47236)
		
	channel1 = grpc.insecure_channel('localhost:50051')
	channel2 = grpc.insecure_channel('localhost:52251')
	stub1 = sgd_pb2_grpc.SGDStub(channel1)
	stub2 = sgd_pb2_grpc.SGDStub(channel2)
	for i in range(1000):
		response = stub1.ComputeTask(sgd_pb2.LWB(labels=labels, weights=weights, batch=batch, corr_1=c1, corr_2=c2))
		update = dict(response.grad_up)
		for x in update:
			weights[x] -= 0.3*update[x]

		print(hinge_loss(labels,batch,weights))







if __name__ == "__main__":
	run()