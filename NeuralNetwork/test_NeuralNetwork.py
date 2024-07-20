import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pdb
import os
import pickle

class ParamInitializer:
	def __init__(self, 
		weight_range=[-.3, .3],
		bias_range=[0,0],
		randomization_method='normal',
		input_layer_size=False):

		self.weight_range = weight_range
		self.method = randomization_method
		self.bias_range = bias_range

		if randomization_method == 'xavier':
			if not input_layer_size:
				raise Exception('xavier method requires specifying input
					layer size')
			self.xavierize(input_layer_size)
		
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

	def pickle(self, name):
		pass

class Function:
	def __init__(self, name : str):
		self.name = name

	def create(self, name: str):
		switch = {
			('Identity','ReLu','Logistic') : lambda n: ActivationFunction(n),
			('onehot', 'identity') : lambda n: EvaluationFunction(n),
			('MeanSquared', 'MeanAbsolute') : lambda n: CostFunction(n)}
		for key_tuple, activator in switch:
			if name in key_tuple:
				return activator(name)
		raise Exception('Function name not recognized')

class ActivationFunction(Function):
	switch = {
		'function' : {
			'ReLu' :     lambda x : x if x > 0 else 0,
			'Logistic' : lambda x : 1 / (1 + math.e**(-x)),
			'Identity' : lambda x : x}, 
		'prime' : {
			'ReLu' :     lambda x : 1 if x > 0 else 0, 
			'Logistic' : lambda x : (math.e**(-x)) / (1 + math.e**(-x))**2,
			'Identity' : lambda x : 1}}

	def vector_activation(self, X, prime_or_func):
		function = self.switch[prime_or_func][self.name]
		return np.array([function(x) for x in X.T.tolist()[0]]).T

	def prime(self, X: (np.array, float, int)) -> (np.ndarray, float):
		if isinstance(X, (float, int)):
			return self.switch['prime'][self.name](X)	
		return self.vector_activation(X, 'prime')

	def function(self, X: (np.array, float, int)) -> (np.ndarray, float):
		if isinstance(X, (float, int)):
			return self.switch['function'][self.name](X)	
		return self.vector_activation(X, 'function')

	def jacobian(self, unactivated_outputs : np.ndarray) -> np.matrix:
		X_m_0 = unactivated_outputs.T.tolist()[0]
		prime = self.switch['prime'][self.name]
		return np.ndarray(np.diag([prime(x) for x in X_m_0]))

class EvaluationFunction(Function):
	def classify(self, Y_hat):
		switch_classify = {
			'onehot': lambda Y_hat : self.one_hot(Y_hat),
			'identity' : lambda Y_hat : Y_hat}
		return switch_classify[self.name](Y_hat)
		
	def evaluate(self, Y : np.ndarray, Y_hat: np.matrix) -> int:
		return int(Y == switch_classify[self.name](Y_hat))
	
	def one_hot(self, Y_hat):
		Y_hat_as_row = Y_hat.reshape(1,-1)
		dim = Y_hat_as_row.shape[0]
		hot_idx = np.argmax(Y_hat_as_row)
		return np.ndarray([(1 if i==hot_idx else 0) for i in range(dim)]).

class CostFunction(Function):
	switch = {
		'function' : {
			'MeanSquared' : np.square,
			'MeanAbsolute' : np.abs},
		'partial' : {
			'MeanSquared' : lambda x : 2 * x,
			'MeanAbsolute' : lambda x : 1 if x >= 0 else 1}}

	def function(self, Y : np.ndarray, Y_hat : np.ndarray) -> np.ndarray:
		error_terms = self.switch['function'][self.name](Y_hat - Y)
		return np.mean(error_terms.T[0])

	def partial(
		self, 
		Y : np.ndarray,
		Y_hat : np.ndarray, 
		del_var_idx : int) -> float:

		m = Y.shape[0]; idx = (del_var_idx, 0)
		x = Y_hat[idx] - Y[idx]
		return self.switch['partial'][self.name](x) / m

	def gradient(self, Y : np.ndarray, Y_hat : np.ndarray):
		return np.ndarray([
			self.partial(Y, Y_hat, del_var_idx)
				for del_var_idx in range(Y.shape[0])])

class Layer:
	def _check_evaluated(self):
		if not self.evaluated:
			raise Exception('Must first call _feedforward method')
		
#	def _prune(self, pruning_map : dict):
#		for input_idx, output_idx in pruning_map.items():
#			self.w_matrix[input_idx][output_idx] = 0 

	def __init__(self, 
		layer_size : int,
		prior_layer : int,
		model : test_NeuralNetwork.Model):
		#pruning_map : (bool, dict) = False,
	
		self.model = model
		self.activation = self.model.activation 
		rand_weight = self.model.initializer.random_weight
		rand_bias = self.model.initializer.random_bias

		m = self.m = self.size = layer_size
		n = self.n = self.input_size = prior_layer

		self.b_vector = np.ndarray(
			[[initializer.random_bias()] 
				for i in range(m)])

		self.w_matrix = np.ndarray(
			[[initializer.random_weight() 
				for j in range(n)]
					for i in range(m)])

		#if pruning_map:
			#self._prune(pruning_map)
		self._clear()
	
	def _clear(self):
		self.input = False 
		self.output = False 
		self.X_m_0 = False 
		self.local_jacobian = False 
		self.bias_updater = False
		self.weight_updater = False
		self.evaluated = False
	
	def XY(self):
		return (self.input, self.output)
	
	def unactivated_outputs(self, input_vector: np.ndarray) -> np.ndarray:
		W, X_n, B = self.w_matrix, input_vector, self.b_vector
		X_m_0 = W @ X_n + B
		return X_m_0
		
	def _feedforward(self, input_vector: np.ndarray) -> np.ndarray:
		self.input = input_vector
		self.X_m_0 = self.unactivated_outputs(self.input)
		self.output = self.activation.function(self.X_m_0)
		activation_jacobian = self.activation.jacobian(self.X_m_0)
		self.local_jacobian = activation_jacobian @ self.w_matrix
		self.evaluated = True
		
	def load_parameter_gradients(self, gradient):
		self._check_evaluated()
		self.weight_gradient = np.ndarray(np.zeros((self.m, self.n)))
		self.bias_gradient = np.ndarray(np.zeros((self.m, 1)))

		def gradient_coefficient(i):
			x_i = self.X_m_0[i,0]
			return gradient[0,i] * self.activation.prime(x_i)

		for i in range(self.m):
			del_Ci = gradient_coefficient(i)
			self.weight_gradient[i] = (del_Ci * self.input).T
			self.bias_gradient[i] = [del_Ci]
	
	def _update(self, learn_rate=.05):
		self._check_evaluated()
		self.w_matrix -= learn_rate * self.weight_gradient
		self.b_vector -= learn_rate * self.bias_gradient
		self._clear()
		
class Model:	
	preloads = ['mnist',
	 'boston_housing',
	 'fashion_mnist',
	 'imdb',
	 'cifar10',
	 'cifar100',
	 'reuters']

	def _onehot(y_data : np.ndarray):
		def onehot_vec(integer, y_rng):
			output = np.zeroes((1, y_rng))
			output[integer - 1] = 1
			return output.reshape(-1, 1)
		return [onehot_vec(y, y_data.max()) for y in y_data]
	
	def _determine_range(vector_list : list):
		return (global_min, global_max)
	
	def _normx(x_data : list):
		maxim = max(vector_list, key= lambda vector : vector.max()).max()
		minim = min(vector_list, key= lambda vector : vector.min()).min()
		normalizer = np.vectorize(lambda x : (x - minim) / (maxim - minim))
		return [normalizer(vector) for vector in x_data]
	
	def _flatten_x(x_data : np.ndarray) -> list:
		return [x.flatten().reshape(-1,1) for x in x_data]
	
	def _preprocess(
		self, 
		raw_data:  tuple, 
		normalize: bool, 
		classify:  bool) -> list:

		train, test = raw_data 

		x_train = self._flatten_x(train[0])
		x_test = self._flatten_x(test[0])
		y_train, y_test = train[1], test[1]

		if normalize: 
			x_train, x_test = self._normx(x_train), self._normx(x_test)
		if classify:
			y_train, y_test = self._onehot(y_train), self._onehot(y_test)
	
		current = (x_train, x_test, y_train, y_test)
		try: 
			old = (self.x_train, self.x_test, self.y_train, self.y_test)
			shape_check = lambda data : data[0].shape[0]
			for curr, old in zip(current, older):
				if not shape_check(curr) == shape_check(old)
					raise Exception('Incompatible dimensions for new data')

		except NameError:
			self.x_train, self.x_test, self.y_train, self.y_test = current

		self.train = (self.x_train, self.y_train)
		self.test = (self.x_test, self.y_test)
		self.data = (self.train, self.test)

		self.input_size = self.x_train[0].shape[0] 
		self.output_size = self.y_train[0].shape[0] 

	def _clear(self):
		self.X, self.Y, self.Y_hat = False, False, False
		self.prediction = False
		for layer in self.inner_layers:
			layer._clear()

	def _update(self, learn_rate=.05):
		for layer in self.inner_layers:
			layer._update(learn_rate=learn_rate)
		self._clear()

		self.gradient_history.setdefault(self.epoch,{})[self.step] = []
		run_gradient = self.cost.gradient(self.Y, self.Y_hat)

		step_gradients = [] 
		for layer in reversed(self.inner_layers):
			step_gradients += [run_gradient]
			layer.load_parameter_gradients(run_gradient)
			run_gradient = run_gradient @ layer.local_jacobian

		step_gradients += [None] 
		#None object corresponds to input layer idx 0
		step_gradients.reverse()
		self.gradient_history[self.epoch][self.step] = step_gradients 

	def _feedforward(self):
		input_ = self.X
		for layer in self.inner_layers:
			layer._feedforward(input_)
			input_ = layer.output

		self.Y_hat = self.inner_layers[-1].output
		self.prediction = self.evaluator.classify(self.Y_hat)
	
		if self.Y == False:
			return None
		return self.cost.function(self.Y, self.Y_hat)
	
	def display_cost(self, trn_tst='train'):
		plt.figure(figsize=(10, 5))
		x = list(self.runcost.keys())
		y = list(self.runcost.values())
		plt.plot(x, y)

		plt.xlabel('Step')
		plt.ylabel('Loss' if trn_test=='train' else 'Acc')

		title = {
			'train' : 'Trailing cost avg at each step of training',
			'test' : 'Trailing accuracy avg at each step of testing'}
		plt.title(title[direction])
		plt.show()
	
	def display_layer_history(self, layer_idx):
		y = self.query_history(layer_slice = layer_idx)
		x = list(range(len(y)))

		plt.figure(figsize=(10, 5))
		plt.plot(x, y)

		plt.xlabel('Total Step Count')
		plt.ylabel('Gradient Magnitude')

		title = f'Gradient Magnitude History for Layer {layer_idx}'
		plt.title(title)
		plt.show()

	def display_step_history(self, epoch, step):
		plt.figure(figsize=(10, 5))

		y = self.query_history(epoch_step_slice=(epoch, step))
		x = list(range(len(y)))

		plt.xlabel('Layer Index')
		plt.ylabel('Gradient Magnitude')

		plt.title(f'GradMag By Layer At Epoch {epoch} Step {step}')
		plt.plot(x, y)
		plt.show()

	def query_history(
		self, 
		layer_slice : (bool, int) = False, 
		epoch_step_slice: (bool, list, tuple) = False):

		if (layer_slice and epoch step_slice):
			raise Exception('Choose only one kwarg at most')
		if not self.gradient_history:
			raise Exception('Network is currently untrained')

		default_output = self.gradient_history 
		output = default_output

		if layer_slice:
			output = []
			for epoch, epoch_dict in self.gradient_history.items():
				for step, step_gradient_list in epoch_dict.items():
					layer_gradient = step_gradient_list[layer_slice]
					output += [np.linalg.norm(layer_gradient)]	

		elif epoch_step_slice:
			e, s = tuple(epoch_step_slice)
			output = list(map(np.linalg.norm, self.gradient_history[e][s]))

		return output 
	
	def train(self, epochs : int=10, learn_rate : float=.05, display=True):
		self._iterate(epochs, learn_rate, display, 'train')
	
	def test(self, display=True):
		self._iterate(1, .0, display, 'test') 
		
	def _iterate(self, 
		epochs : int, 
		lrn_rt : float, 
		disp: bool, 
		trn_tst: str):

		self.runcost = {}
		cumulative_error, total_steps = 0, 0

		X_array, Y_array = self.train if trn_tst=='train' else self.test

		for epoch in range(epochs):
			self.epoch = epoch
			for step, XY in enumerate(zip(X_array, Y_array)):
				self.X, self.Y = XY
				self.step = step
				step_loss = self._feedforward()
				if trn_tst == 'train':
					self._backpropagate()
					self._update()
				elif trn_test == 'test':
					step_loss = (self.prediction == self.Y)
				cumulative_value += step_loss
				total_steps += 1
				self.runcost[total_steps] = cumulative_value / total_steps
				self._clear()

		if disp:
			self.display_cost(trn_test=trn_tst)

	def predict(self, X, clear=True):
		self.X = X
		self._feedforward()
		prediction = self.prediction
		if clear:
			self._clear()
		return prediction

class NeuralNetwork(Model):
	hyperparameters = {
		'initializer' : ParamInitializer(),
		'activation' : ActivationFunction('ReLu'),
		'evaluator' : EvaluationFunction('onehot'),
		'cost' : CostFunction('MeanSquared')}

	def __init__(
		self,
		raw_data: tuple,
		hidden_layer_sizes: list,
		normalize: bool,
		classify: bool,
		**kwargs):
	
		self._preprocess(raw_data, normalize, classify)
		self.hidden_layer_sizes = hidden_layer_sizes
		self.inner_layer_sizes = hidden_layer_sizes + [self.output_size]
		self.all_layer_sizes = [self.input_size] + self.inner_layer_sizes

		for key, value in kwargs.items():
			if not kwarg in self.hyperparameters:
				raise Exception('Unrecognized kwarg in __init__')
			if value == False:
				kwargs.pop(key)
		self.hyperparameters.update(kwargs)
		self.__dict__.update(self.hyperparameters)
		#self._sanitize_pruner_list()

		self.inner_layers = []
		input_size = self.input_size
		for i, output_size in enumerate(self.inner_layer_sizes):
			layer = Layer(output_size, input_size, self) #self.pruner_list[i]
			self.inner_layers += [layer]
			input_size = output_size

		self.depth = len(self.inner_layers)
		self.gradient_history = dict()

	def _load_raw_data(self, data_str):
		if data.endswith('.pkl'):
			with open(data, 'rb') as file:
				raw_data = pickle.load(file)
		elif data in self.preloads:
			from tensorflow.keras import datasets	
			raw_data = datasets.__dict__[data].load_data()
		else:
			raise Exception('Invalid data call')
		return raw_data

class ParamPublic(ParamInitializer):
	def __init__(self, args):
		a = args
		super.__init__(
			weight_range = a.weight_range,
			bias_range = a.bias_range,
			randomization_method = a.method,
			input_layer_size = a.input_layer_size)
			
class NeuralPublic(NeuralNetwork):
	def __init__(self, args):
		a = args
		raw_data = self._load_raw_data(a.data)
		super().__init__(
			raw_data,
			a.hidden,
			a.normalize,
			a.classify,
			initializer = a.param_initializer,
			activation = a.activation,
			evaluator = a.evaluation,
			cost = a.cost)
		
def random_y(output_size):
	j = random.randint(0, output_size - 1)
	return [1 if i == j else 0 for i in range(10)]

def random_x(input_size, mu=0, sigma=1):
	return [random.normalvariate(mu, sigma) for i in range(input_size)]

@pytest.fixture
def hidden_layer_sizes():
	return [256, 128]
	
def random_img_vector(image_height, image_width):
	return np.array([[random.randint(0, 255) for i in range(image_width)]
			for j in range(image_height], dtype=np.uint8])

def random_vector_uniform(input_size, lower=0, upper=1):
	return [random.uniform(lower, upper) for i in range(input_size)]

@pytest.fixture
def image_height():
	return 28 

@pytest.fixture
def image_width():
	return 28

@pytest.fixture
def dataset_size():
	return 100 

@pytest.fixture
def input_size(image_height, image_width):
	return image_height * image_width

@pytest.fixture
def output_size():
	return 10
	
@pytest.fixture
def X_data_random(image_height, image_width, dataset_size):
	return np.array([random_img_vector(image_height, image_width)
			for j in range(dataset_size)])

@pytest.fixture
def Y_data_random(output_size, dataset_size):
	d_rng = range(dataset_size)
	return np.array([random.randint(0, output_size-1) for i in d_rng])
		
@pytest.fixture
def mnist_mock(X_data_random, Y_data_random, dataset_size):
	split = dataset_size // 5
	X_train, X_test = X_data_random[split:], X_data_random[:split]
	Y_train, Y_test = Y_data_random[split:], Y_data_random[:split]
	train = (X_train, Y_train)
	test = (X_test, Y_test)
	return (train, test)

@pytest.fixture
def grawp(hidden_layer_sizes, mnist_mock):
	return NeuralNetwork(mnist_mock, hidden_layer_sizes, True, True)

def test_backpropagate(grawp, X_data_random, Y_data_random):
	grawp.train()
	for X, Y in zip(X_data_random, Y_data_random):
		prediction = grawp.predict(X) 
		pdb.set_trace()
		Y = np.ndarray([Y]).T
		assert prediction.all() != Y.all()
	
def test_predict(grawp):
	pass

