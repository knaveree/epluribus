import random 
import math
from math import sqrt
import numpy as np

class Layer:
	def __init__(
		input_size,
		output_size,
		weight_method='He', 
		activation_method='ReLU'):
	
		self.weight_method = weight_method
		self.activation_method = activation_method

		self.B = np.array([
			[self.bias_init() for i in range(input_size)]])
		self.W = np.array([
			[self.weight_init() for i in range(input_size)]
				for j in range(output_size)])

	def column(self, vectorable):
		if not isinstance(vectorable, np.ndarray):
			vectorable = np.ndarray(vectorable)
		return vectorable.reshape(-1, 1)

	def scalar_activation(self, x, prime=False):
		if self.activation_method=='ReLU':
			if prime:
				return 0 if x < 0 else 1
			return 0 if x < 0 else x
		else:
			raise Exception('Unimplemented activation method')

	def weight_init(self):
		n = self.n
		m = self.m

		if self.weight_method == 'he':
			return random.normalvariate(0.0, sqrt(2/n))
		if self.weight_method == 'xavier':
			x = 1 / sqrt(n)
			return random.uniform(-x,x)
		if self.weight_method == 'norm_xavier':
			x = sqrt(6) / sqrt(n + m)
			return random.uniform(-x, x)
	
	def bias_init(self):
		return 0.0

	def activation(self, Z_X, jacobian=False):
		if jacobian:
			activation_prime_vectorized = np.vectorize(self.scalar_activation)
			activation_prime_evaluated = activation_prime_vectorized(Z_X)
			return np.diag(activation_prime_evaluated)
		else:
			return self.column(np.vectorize(self.scalar_activation)(Z_X))
		
	def Z(self, X):
		return self.column(self.W @ X + self.B)

	def feedforward(self, X):
		self.Z_X = self.Z(X)
		self.X_out = self.activation(self.Z(X))
	
	def backprop(self, incoming_gradient, X):
		incoming_gradient = incoming_gradient.reshape(1,-1)
		activation_jacobian = self.activation(self.Z_X, jacobian=True)
		self.bias_updater = incoming_gradient @ activation_jacobian

		#The bias updater also happens to be 'complete' incoming gradient
		#from the perspective of the preactivation function, thus serving 
		#as one of the vectors for the weight updater outer product,
		#as well as the outgoing gradient

		self.weight_updater = np.outer(self.bias_updater, X)
		self.outgoing_gradient = self.bias_updater @ self.W
	
	def update(self, alpha):
		self.W = self.W - alpha * self.weight_updater
		self.B = self.B - alpha * self.bias_updater
			
class Network:
	def __init__(
		self, 
		architecture,
		weight_method='He',
		activation_method='ReLU',
		evaluation='OneHot',
		cost_function='MeanSquared'):

		self.evaluation = evaluation
		self.cost_function = cost_function

		self.layers = []
		arch = architecture
		for input_size, output_size in zip(arch[:-1], arch[1:]):
			layers += Layer(
				input_size, 
				output_size, 
				weight_method = weight_method 
				activation_method = activation_method)

		self.n, self.m = arch[0], arch[-1]
		self.evaluated = False	
	
	def vector(self, vectorable):
		return np.array(vectorable).reshape(-1,1)
	
	def loss(self, Y):
		Y = self.vector(Y)	
		Y_hat = self.vector(self.Y_hat)
		if self.cost_function == 'MeanSquared':
			return np.sum((Y - Y_hat)**2 / self.m)
		else:
			raise Exception('Undefined cost function')
	
	def cost_gradient(self, Y, Y_hat):
		if self.cost_function == 'MeanSquared':
			n = self.layers[-1].m
			return (2 / n) * (Y_hat - self.vector(Y)) 
		else:
			raise Exception('Unrecognized cost function')

	def feedforward(self, X, Y):
		for layer in self.layers:
			layer.feedforward(X)
			X = layer.X_out
		self.Y_hat = layers[-1].X_out
		self.evaluated = True

	def evaluate(self, Y_hat):
		if self.evaluation == 'OneHot':
			return self.column([1 if k == np.argmax(Y_hat) else 0 
				for k in range(len(Y_hat))])
		else:
			raise Exception('Undefined Evaluation Function')

	def predict(self):
		if not self.evaluated:
			self.feedforward()
		return self.evaluate(Y_hat)

	def backprop(self, X, Y):
		self.feedforward(X, Y)
		incoming_gradient = self.cost_gradient(Y, self.Y_hat)

		inner_reversed_layers = list(reversed(self.layers[1:]))
		for l, layer in enumerate(inner_reversed_layers):
			X_in = reversed_layers[l+1].X_out
			layer.backprop(incoming_gradient, X_in)
			incoming_gradient = layer.outgoing_gradient
		layer[0].backprop(incoming_gradient, X)
	
	def update(self):
		for layer in self.layers:
			layer.update()

	def train(X_data, Y_data, alpha=.05, epochs=10):	
		self.run_loss = {}	
		for epoch in epochs:
			self.runloss[epoch] = []
			for X, Y in zip(X_data, Y_data):
				self.backprop(X, Y)
				self.runloss[epoch] += self.loss(Y)
				self.update(alpha=alpha)

	def unpack_runloss(self):	
		unpacked = []
		for epoch in self.runloss:
			unpacked += self.runloss[epoch]
		return unpacked
		
