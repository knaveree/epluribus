import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pdb

class ParamInitializer:
	def __init__(self, 
		weight_range=(-.3, .3),
		bias_range=(0,0),
		randomization_method='uniform',
		input_layer_size=None,
		preset_package = False):

		self.weight_range = weight_range
		self.method = randomization_method
		self.bias_range = bias_range

		if randomization_method == 'xavier':
			self.xavierize(input_layer_size)
		if preset_package:
			self.__dict__.update(preset_package)
		
	def xavierize(self, input_layer_size):
		n = input_layer_size
		self.weight_range = (-1/math.sqrt(n), 1/math.sqrt(n))
		self.method = 'uniform'

	def normrandom(self, range_tuple):
		stddev = (range_tuple[1] - range_tuple[0]) / 2
		mean = range_tuple[0] + stddev
		return random.normalvariate(mean, stddev)
	
	def unirandom(self, range_tuple):
		low, high = range_tuple
		return random.uniform(low, high)

	def random_weight(self):
		if self.method == 'uniform':
			return self.unirandom(self.weight_range)
		elif self.method == 'normal':
			return self.normrandom(self.weight_range)

	def random_bias(self):
		if self.method == 'uniform':
			return self.unirandom(self.bias_range)
		elif self.method == 'normal':
			return self.normrandom(self.bias_range)

class Function:
	def __init__(self, name : str):
		self.name = name
	
	def vectorized_function(self, X, function):
		output = np.zeros(X.shape[0])
		for i, x in enumerate(X.T.tolist()[0]):
			output[i] = function(x)
		return np.matrix(output).T

class ActivationFunction(Function):
	prime_switch = {
		'ReLu' : lambda x : 1 if x > 0 else 0, 
		'Logistic' : lambda x : (math.e**(-x)) / (1 + math.e**(-x))**2,
		'Identity' : lambda x : 1}
	
	function_switch = {
		'ReLu' : lambda x : x if x > 0 else 0,
		'Logistic' : lambda x : 1 / (1 + math.e**(-x)),
		'Identity' : lambda x : x}

	def prime(self, X: np.matrix) -> np.matrix:
		function = self.prime_switch[self.name]
		return self.vectorized_function(X, function)

	def function(self, X: np.matrix) -> np.matrix:
		function = self.function_switch[self.name]
		return self.vectorized_function(X, function)

	def jacobian(self, unactivated_outputs : np.matrix):
		X_m_0 = unactivated_outputs.T.tolist()[0]
		prime = self.prime_switch[self.name]
		return np.matrix(np.diag([prime(x) for x in X_m_0]))

class EvaluationFunction(Function):
	def classify(self, Y_hat):
		switch_classify = {
			'OneHot': lambda Y_hat : self.one_hot(Y_hat),
			'Identity' : lambda Y_hat : Y_hat}
		return switch_classify[self.name](Y_hat)
		
	def evaluate(self, Y : np.matrix, Y_hat: np.matrix) -> int:
		return int(Y == switch_classify[self.name](Y_hat))
	
	def one_hot(self, Y_hat):
		return np.matrix([
			1 if i==np.argmax(Y_hat.T[0]) else 0 
				for i in range(Y_hat.shape[0])]).T

class CostFunction(Function):
	def function(
		self, 
		Y : (list, np.matrix),
		Y_hat : np.matrix):

		switch_case = {
			'MeanSquared' : self.mean_squared,
			'MeanAbsolute' : self.mean_absolute,
			'CrossEntropyLoss' : self.cross_entropy}
		return switch_case[self.name](Y, Y_hat)

	def partial(
		self, 
		Y : (list, np.matrix), 
		Y_hat : np.matrix, 
		del_var_idx : int) -> float:

		switch_case = {
			'MeanSquared' : self.mean_squared_partial,
			'MeanAbsolute' : self.mean_absolute_partial,
			'CrossEntropyLoss' : self.cross_entropy_partial}
		return switch_case[self.name](Y, Y_hat, del_var_idx)

	def mean_squared(self, Y, Y_hat):
		error_terms = np.square(Y - Y_hat)
		return np.mean(error_terms.T[0])

	def	mean_squared_partial(self, Y, Y_hat, del_var_idx):
		m=len(Y.shape[0]); i=del_var_idx,
		return (Y_hat[i][0] - Y[i][0]) * (2 / m)
	
	def gradient(self, Y, Y_hat):
		return np.matrix([
			self.partial(Y, Y_hat, del_var_idx)
				for del_var_idx in range(len(Y.shape[0]))])

class Layer:
	def _prune(self, pruning_map : dict):
		for input_idx, output_idx in pruning_map.items():
			self.w_matrix[input_idx][output_idx] = 0 

	def __init__(self, 
		layer_size : int,
		prior_layer : int,
		initializer : ParamInitializer, 
		pruning_map : (bool, dict) = False,
		activation = ActivationFunction('ReLu')):
	
		self.activation = activation 
		rand_weight = initializer.random_weight
		rand_bias = initializer.random_bias

		m = self.m = self.size = layer_size
		n = self.n = self.input_size = prior_layer

		self.b_vector = np.matrix(
			[[initializer.random_bias()] 
				for i in range(m)])

		self.w_matrix = np.matrix(
			[[initializer.random_weight() 
				for j in range(n)]
					for i in range(m)])

		if pruning_map:
			self._prune(pruning_map)
		self._clear()
	
	def _clear(self):
		self.input = False 
		self.output = False 
		self.XMO = False 
		self.local_jacobian = False 
		self.bias_updater = False
		self.weight_updater = False
		self.evaluated = False
	
	def XY(self):
		return (self.input, self.output)
	
	def unactivated_outputs(self, input_vector: np.matrix) -> np.matrix:
		W, X_n, B = self.w_matrix, input_vector, self.b_vector
		X_m_0 = W * X_n + B
		return X_m_0
		
	def _feedforward(self, input_vector: np.matrix) -> np.matrix:
		self.input = input_vector
		self.X_m_0 = self.unactivated_outputs(self.input)
		self.output = self.activation.function(self.X_m_0)
		activation_jacobian = self.activation.jacobian(self.X_m_0)
		self.local_jacobian = activation_jacobian * self.w_matrix
		self.evaluated = True
		
	def load_parameter_gradients(self, gradient):
		if not self.evaluated:
			raise Exception('Must first call _feedforward method')

		#We are abusing the notation a bit here; the weight and bias
		#"gradients" should be row vectors but we're reorganizing them
		#for computational convenience; furthermore they're actually
		#just slices of a much larger actual gradient rather than gradients
		#in their own right. You probably don't care but, just sayin'.

		self.weight_gradient = np.matrix(np.zeros(self.m, self.n))
		self.bias_gradient = np.matrix(np.zeros(self.m, 1))

		def gradient_coefficient(i):
			del_C_del_sigma_i = gradient[0][i] 
			del_sigma_i_del_Xm0_i = self.activation.prime(self.X_m_0[i][0])
			return del_C_del_sigma_i * del_C_del_Xm0_i

		for i in range(self.m):
			del_Ci = gradient_coefficient(i)
			self.weight_gradient[i] = (del_Ci * self.input).T
			self.bias_gradient[i] = [del_Ci]
			
class Model:
	def _vectorize(self, inputs : (list, np.matrix)):
		if isinstance(inputs, list):
			return np.matrix(inputs).T	
		elif isinstance(inputs, np.matrix):
			if inputs.shape[0] == 1:
				return inputs.T
			if inputs.shape[1] == 1:
				return inputs
			else:
				raise Exception('Invalid input')
		else:
			raise Exception('Invalid input')

	def _reversed_inner_layers(self):
		return list(reversed(self.inner_layers))

	def _clear(self):
		self.X, self.Y, self.Y_hat = False, False, False
		self.prediction = False
		for layer in self.inner_layers:
			layer._clear()

	def _update(self, learn_rate=.05):
		for layer in self.inner_layers:
			layer.update(learn_rate=learn_rate)
		self._clear()

	def _backpropagate(self):
		cum_gradient = self.cost.gradient(self.Y, self.Y_hat)
		self.gradient_history.setdefault(self.epoch,{})[self.step] = []

		for layer in self.reversed_inner_layers():
			self.gradient_history[self.epoch][self.step] += [cum_gradient]
			layer.load_parameter_gradients(cum_gradient)
			cum_gradient *= layer.local_jacobian

	def _feedforward(
		self, 
		X: (list, np.matrix), 
		Y: (list, np.matrix, bool) = False) -> (float, None):

		pdb.set_trace()
		self.X = self._vectorize(X)
		if isinstance(Y, (list, np.matrix)):
			self.Y = self._vectorize(Y)
		else: 
			self.Y = False

		input_ = self.X
		for layer in self.inner_layers:
			layer._feedforward(input_)
			input_ = layer.output

		self.Y_hat = self.inner_layers[-1].output
		self.prediction = self.evaluator.classify(self.Y_hat)

		return self.cost.function(self.Y, self.Y_hat)
	
	def display_cost(self, direction='train'):
		plt.figure(figsize=(10, 5))
		plt.plot(list(range(len(self.runcost))), self.runcost)

		plt.xlabel('Step')
		plt.ylabel('Loss' if direction=='train' else 'Acc')

		title = {
			'train' : 'Trailing cost avg at each step of training',
			'test' : 'Trailing accuracy avg at each step of testing'}
		plt.title(title[direction])
		plt.show()
	
	def train(
		self, 
		X_train : list, 
		Y_train : list, 
		epochs : int=10, 
		learn_rate : float=.05, 
		display=True):

		self._iterate(X_train, Y_train, epochs, learn_rate, display, 'train')
	
	def test(
		self, 
		X_test : list, 
		Y_test : list, 
		display=True):

		self._iterate(X_test, Y_test, 1, .0, display, 'test') 
		
	def _iterate(self, 
		X_array : list, 
		Y_array : list, 
		epochs : int, 
		learn_rate : float,
		display: bool,
		direction: str):

		if not len(X_array) == len(Y_array):
			raise Exception('Mismatched X and Y lengths')

		self.runcost = []
		cumulative_error, total_steps = 0, 0

		for epoch in range(epochs):
			self.epoch = epoch
			for step, XY in enumerate(zip(X_array, Y_array)):
				X, Y = XY
				self.step = step
				step_loss = self._feedforward(X, Y=Y)
				if direction == 'train':
					self._backpropagate()
					self._update()
				elif direction == 'test':
					step_loss = (self.prediction == self.Y)
				cumulative_error += step_loss
				total_steps += 1
				self.runcost[total_steps] = cumulative_error / total_steps
				self._clear()

		if display:
			self.display_cost(direction=direction)

	def predict(self, X, clear=True):
		self._feedforward(X)
		prediction = self.prediction
		if clear:
			self._clear()
		return prediction

	def query_history(self):
		if not self.gradient_history:
			print ('Network is currently untrained')
		return self.gradient_history

class LinearModel(Model):
	initializer = ParamInitializer(
		weight_range = (-1,1),
		bias_range = (0,0),
		randomization_method = 'normal')

	hyperparameters = {
		'initializer' : initializer,
		'activation' : ActivationFunction('Identity'),
		'evaluator' : EvaluationFunction('Identity'),
		'cost' : CostFunction('MeanSquared'),
		'pruner_list' : False}

	def __init__(self):
		return None

class NeuralNetwork(Model):
	initializer = ParamInitializer(
		weight_range = (-.3,3),
		bias_range = (0,0),
		randomization_method = 'uniform')

	hyperparameters = {
		'initializer' : initializer,
		'activation' : ActivationFunction('ReLu'),
		'evaluator' : EvaluationFunction('OneHot'),
		'cost' : CostFunction('MeanSquared'),
		'pruner_list' : False}

	def _sanitize_pruner_list(self):
		if self.pruner_list:
			warning = 'pruner_list does not match architecture'
			match = len(self.pruner_list) == len(self.inner_layers)

			if not match:
				raise Exception(warning)

			for i, pruning_map in enumerate(self.pruner_list): 
				input_size = self.all_layer_sizes[i]
				output_size = self.all_layer_sizes[i+1]
				for i, j in pruning_map.items():
					if not i < input_size and j < output_size:
						raise Exception(warning)
		else:
			self.pruner_list = [
				False for layer in self.inner_layer_sizes]

	def __init__(
		self,
		input_size: int,
		hidden_layer_sizes: list,
		output_size : int,
		**kwargs):
		
		self.hidden_layer_sizes = hidden_layer_sizes
		self.input_size = input_size
		self.output_size = output_size
		self.inner_layer_sizes = hidden_layer_sizes + [output_size]
		self.all_layer_sizes = [input_size] + self.inner_layer_sizes

		for kwarg in kwargs:
			if not kwarg in self.hyperparameters:
				raise Exception('Unrecognized kwarg in __init__')
		self.hyperparameters.update(kwargs)
		self.__dict__.update(self.hyperparameters)
		self._sanitize_pruner_list()

		self.inner_layers = []
		input_size = self.input_size
		for i, output_size in enumerate(self.inner_layer_sizes):
			layer = Layer(
				output_size,
				input_size,
				self.initializer,
				self.pruner_list[i],
				activation = self.activation)
			self.inner_layers += [layer]
			input_size = output_size

		self.depth = len(self.inner_layers)
		self.gradient_history = dict()
	
@pytest.fixture
def seed():
	random.seed(10)

@pytest.fixture
def rwrg(seed):
	lb = random.uniform(-1, 0)
	return (lb, +lb)	

@pytest.fixture
def swrg(seed):
	return (-.3, .3)

@pytest.fixture
def input_size():
	return 784

@pytest.fixture
def RUInitializer(rwrg):
	return ParamInitializer(
		weight_range=rwrg,
		randomization_method='uniform')

@pytest.fixture
def RXInitializer(rwrg, input_size):
	return ParamInitializer(
		weight_range=rwrg,
		randomization_method='xavier',
		input_layer_size=input_size)

@pytest.fixture
def RNInitializer(rwrg):
	return ParamInitializer(
		weight_range=rwrg,
		randomization_method='normal')

def test_RUInitializer(rwrg, RUInitializer):
	for i in range(20):	
		assert rwrg[0] <= RUInitializer.random_weight() <= rwrg[1]
		assert RUInitializer.random_bias() == 0

def test_RNInitializer(rwrg, RNInitializer):
	for i in range(100):	
		one_std_out = 0
		if rwrg[0] <= RNInitializer.random_weight() <= rwrg[1]:
			one_std_out += 1	
		assert RNInitializer.random_bias() == 0
	if one_std_out > 50:
		raise Exception('Either unlikely event happened or we have a problem')

def test_RXInitializer(input_size, RXInitializer):
	n = input_size
	wrg = (-math.sqrt(n), math.sqrt(n))
	for i in range(20):
		assert wrg[0] <= RXInitializer.random_weight() <= wrg[1]

@pytest.fixture
def randx():
	return [random.normalvariate(0, 10) for i in range(10)]

@pytest.fixture
def randy():
	j = random.randint(0,9)
	return [1 if i == j else 0 for i in range(10)]

@pytest.fixture
def x_array():
	return [[random.normalvariate(0, 10) for i in range(10)]
			for j in range(10)]

@pytest.fixture
def y_array():
	return [[random.normalvariate(0, 10) for i in range(10)]
			for j in range(10)]

@pytest.fixture
def grawp():
	grawp = NeuralNetwork(
		10,
		[10],
		10)
	return grawp

def test_predict(grawp, randx, randy):
	y_hat = grawp.predict(randx)
	assert y_hat.shape == randy.shape

def test_backpropagate(grawp, x_array, y_array):
	grawp.train(x_array, y_array, epochs=10, learn_rate=0.5, display=True)
	for X, Y in zip(x_array, y_array):
		Y_hat = grawp.predict(X) 
		assert Y_hat != Y
	
