from concurrent import futures
import time

import grpc
import sgd_pb2_grpc
import sgd_pb2
import sys
import numpy as np
#def compute_gradient(y, X, w):
from concurrent import futures



_ONE_DAY_IN_SECONDS = 60 * 60 * 24



def compute_grad(y, X, w,c1,c2, lambda_):
	def is_support(y_n, x_n, w):
		return y_n*sum([x_n[x]*w[x] for x in x_n])<1
	res = X.copy()
	if is_support(y,X,w):
		for x in X:
			if y == 1:
				res[x] = -X[x] * y * c2 + lambda_ * w[x]
			else:
				res[x] = -X[x] * y * c1 + lambda_ * w[x]
	else:
		for x in X:
			res[x] = lambda_ * w[x]
	return res




class SGDServicer(sgd_pb2_grpc.SGDServicer):

	def ComputeTask(self, request, context):
		labels = request.labels
		weights = request.weights
		batch = request.batch
		corr_1 = request.corr_1
		corr_2 = request.corr_2
		lambda_ = request.lambda_

		return sgd_pb2.Update(grad_up=compute_grad(labels, dict(batch), weights,corr_1,corr_2, lambda_))


def serve(port):
	max_workers = 20
	print("Connecting to port", port)
	print("Number of workers", max_workers)
	server = grpc.server(futures.ThreadPoolExecutor (max_workers=max_workers))
	sgd_pb2_grpc.add_SGDServicer_to_server(SGDServicer(), server)
	server.add_insecure_port('[::]:'+str(port))
	server.start()
	try:
		while True:
			time.sleep(_ONE_DAY_IN_SECONDS)
	except KeyboardInterrupt:
		server.stop(0)


if __name__ == '__main__':
	serve(int(sys.argv[1]))
		

