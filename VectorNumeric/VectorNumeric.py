#So we want vector functions here that can be composed and differentiated once. #Thus when we define a function we just need to define all its partial 
#derivatives for each input. 
import numpy as np

class ScalarFunc:
	'''A class for a scalar valued multivariate differentiable function.'''
	def sanitize_evaluator_and_partials(self):
		test = np.array([.0 for i in range(self.input_dimension)])
		try:
			test_x = self.evaluator(test)
			test_grad = self.gradient(test)
		except:
			raise Exception('Dim mismatch in evaluator or partials')
	
	def sanitize(self, vector, do_reshape):
		if do_reshape:
			vector = np.array(vector).reshape(-1, 1)
		if not vector.shape == (self.input_dimension, 1):
			raise Exception('Invalid input dimensions')
		return vector

	def __init__(self, evaluator : lambda, partials : list):
		'''Evaluator should be a lambda that accepts a column vector of 
		length precisely the length of the list of partial derivatives''')
		self.sanitize_evaluator_and_partials()
		self.partials = partials
		self.evaluator = evaluator
		self.input_dimension = len(partials)
	
	def gradient(self, vector, do_reshape=False):
		vector = self.sanitize(vector, do_reshape)
		return np.array([partial(vector) for partial in self.partials])

	def __call__(self, vector, do_reshape=False):
		vector = self.sanitize(vector, do_reshape)
		return self.evaluator(vector)
	
class LinearFunctional(ScalarFunc):
	'''A class for linear functionals that can be initialized from
	simply a numpy row array or a list of numbers'''

	def sanitize(self, entries, do_reshape):
		if isinstance(entries, list):
			entries = np.array(entries)
		if not entries.shape[0] == 1:
			if do_reshape:
				gradient = entries.reshape(1,-1)
			else:
				raise Exception('Invalid np.ndarray dimensions')
		return entries 

	def __init__(self, entries : (list, np.ndarray), do_reshape=False):
		gradient = self.sanitize(entries, do_reshape)	
		evaluator = lambda vector_input : (gradient @ vector_input)[0]
		partials = [(lambda x_i : partial) for partial in gradient]
		super().__init__(evaluator, partials)
	
class VectorFunction:
	def set_input_dimension(self, scalar_funclist):
		input_dimension = None
		for scalarfunc in scalar_funclist:
			curr_input_dimension = scalarfunc.input_dimension
			if input_dimension == None:
				input_dimension = curr_input_dimension
			elif not input_dimension == curr_input_dimension:
				raise Exception('Mismatched dimensions for VectorFunction')
		self.input_dimension = input_dimension
	
	def __init__(self, scalar_funclist : (list, np.ndarray)):
		self.set_input_dimension(scalar_funclist)
		self.composition_list = [scalar_funclist]
		self.output_dimension = len(scalar_funclist)

	def compose(self, other):
		composition_list = self.composition_list + other.composition_list
		return VectorFunction(composition_list)
	
	def __call__(self, vector):
		for scalar_funclist in reversed(self.composition_list):
			vector = np.array(
				[scalarfunc(vector) for scalarfunc in scalar_funclist]
					).reshape(-1,1)
		if vector.shape == (1,1):
			return vector[0]
		return vector
	
	def jacobian(self, vector):
		for scalar_funclist in self.composition_list:
			jacobian = np.array([scalarfunc.gradient(vector) 
				for scalarfunc in scalar_funclist])
			vector = jacobian @ vector
		return jacobian

	def gradient(self, vector):
		if self.output_dimension == 1:
			return self.jacobian(vector).reshape(1,-1)
		else:
			raise Exception('Vector valued function has no unique gradient')

class Eye(LinearMap):
	def __init__(self, dimension : int):
		super().__init__(np.eye(dimension))	
		
class LinearMap(VectorFunction):
	def __init__(self, matrix : np.ndarray):
		self.matrix = matrix
		super().__init__([LinearFunctional(row) for row in matrix])

	def __call__(self, vector):
		if not vector.shape[1]==1:
			raise Exception('Invalid input to LinearMap')
		return self.matrix @ vector
	
	def __rmatmul(self, matrix : np.ndarray):
		return LinearMap(matrix @ self.matrix)

	def __matmul__(self, right_matrix):
		if isinstance(right_matrix, LinearMap):
			right_matrix = linear.matrix
		return LinearMap(self.matrix @ right_matrix)
	
	def jacobian(self):
		return self.matrix

class Pairwise(VectorFunction):
	def sanitize_binary_funcs(self, *args):
		vector_a, vector_b, idx = np.zeros((1,1)), np.zeros((1,1)), 0
		for func in args:
			try:
				x = func(vector_a, vector_b, idx)	
			except: 
				raise Exception(
					'Callable func should take two vectors and index')
			if not isinstance(x, (float, int)):
				raise Exception(
					'Output of func should be float or int')	

	def __init__(self, fixed_vector, binary_func, binary_prime): 
		fix = fixed_vector
		self.sanitize_binary_funcs(func, prime)		

		evaluator_list = [
			lambda v : binary_func(fix, v, i) 
				for i in range(len(fix))]
		partials_list = [
			[lambda v : binary_prime(fix, v, i) 
				if i==j else lambda v : 0 for i in range(len(fix))] 
					for j in range(len(fix))]
		scalar_funclist = [ScalarFunc(evaluator, partials) for 
			evaluator, partials in zip(evaluator_list, partials_list)]

		super().__init__(scalar_funclist)

class FixedSum(Pairwise):
	def __init__(self, fixed_vector)
		binary_func = lambda fix, v, i : fix[i] + v[i]
		binary_prime= lambda fix, v, i : 1
		super().__init__(self, fixed_vector, binary_func, binary_prime)
			
class ElementWise(Pairwise):
	def __init__(
		self, 
		func : callable,
		prime : callable,
		dimension : int): 
		
		self.sanitize_unitary_funcs(func, prime)
		fixed_vector = np.zeros((dimension, 1))

		#Wrapping unitary functions in binary functions to feed through
		#binary function class
		binary_func = lambda fixed_vector, v, i : func(v, i)  
		binary_prime = lambda fixed_vector, v, i : prime(v, i)

		super().__init__(fixed_vector, binary_func, binary_prime)

class CostFunction(ScalarFunction):
	library = {
		'MeanSquared' : {
			'evaluator' : (lambda Y : 
				lambda Y_hat : np.mean((Y - Y_hat)**2)),
			'partials' :  (lambda Y :
				lambda Y_hat : [(lambda Y_hat, i :
					2 * (Y_hat[i] - Y[i]) / len(Y) for i in range(len(Y)))])},
		'MeanAbsolute' : {
			'evaluator' : (lambda Y : 
				lambda Y_hat : np.mean(np.abs(Y - Y_hat))),
			'partials' :  (lambda Y :
				lambda Y_hat : [(lambda Y_hat, i :
					(-1 if (Y_[i] - Y_hat[i]) >= 0 else 1) / len(Y) 
						for i in range(len(Y)))])}
	}

	def __init__(self, Y, name):
		evaluator = library[name]['evaluator']
		partials  = library[name]['partials']
		super().__init__(evaluator, partials)

class ActivationFunction(Elementwise):
	library = {
		'ReLu' : { 
			'func' : lambda x : x if x > 0 else 0,
			'prime': lambda x : 1 if x > 0 else 0}
		'Logistic' : 
			'func' : lambda x : 1 / (1 + math.e**-x),
			'prime': lambda x : math.e**(-x) / (1 + math.e**(-x))**2
	}

	def __init__(self, name, dimension):
		func = library.get(name)['func']
		prime = library.get(name)['prime']
		super().__init__(func, prime, dimension)

class EvaluationFunction:
	library = {
		'OneHot'   : lambda v : np.array(
			[1 if i==np.argmax(v) else 0 
				for i in range(v.shape[0])]),
		'Identity' : lambda v : v}

	def __init__(self, name, dimension):
		self.name = name	
		self.dimension = dimension
	
	def __call__(self, vector):
		vector = vector.reshape(-1,1)	
		if not vector.shape[0] == self.dimension:
			raise Exception('Invalid vector input dimensions')
		return self.library[self.name](vector)

class NeuralNetwork:
	hyperparameters = {
		'w_rng' 		: (-.3,.3),
		'b_rng'   		: (.0,.0),
		'rand_method'  	: 'uniform',
		'epochs'  		: 10,
		'hidden'  		: [100, 50],
		'cost'			: 'MeanSquared',
		'activation' 	: 'ReLu', 
		'evaluation'	: 'OneHot'}
		'learn_rate'	: .05}
	
	def layer_output_dimension(self, layer_idx):
		return self.weights[layer_idx].shape([0])
	
	def layer_input_dimension(self, layer_idx):
		return self.weights[layer_idx]).shape([1])

	def randomizer(self, param_type):
		low, high = self.w_rng if param_type == 'w' else self.b_rng
		if self.rand_method == 'xavier' and param_type == 'w':
			low, high = -1 / self.input_size, 1 / self.input_size
		if self.rand_method in ('xavier', 'uniform'):
			return random.uniform(low, high)
		elif self.rand_method == 'normal':
			stddev = (high - low) / 2
			mean = low + stddev
			return random.normalvariate(mean, stddev)
		else:
			raise Exception('Invalid randomization method')
	
	def int_y_arrays_to_onehot(self):
		name_array_zip = zip(
			('train_y', 'test_y'), (self.train_y, self.test_y))
		for train_or_test, int_array in name_array_zip:
			dim_range = lambda array : range(max(array) + 1)
			Y_as_onehot = np.array([
				[1 if i==integer else 0 for i in dim_range(int_array)]
					for integer in int_array])
			self.__dict__[name] = Y_as_onehot
	
	def flatten_x(self):
		name_array_zip = zip(
			('train_x', 'test_x'), (self.train_x, self.test_x))
		for train_or_test, x_array in name_array_zip:
			X_flattened = np.array([x.flatten() for x in x_array]) 
			self.__dict__[name] = X_flattened
	
	def sanitize_data(train_test_split, x_flatten=True, y_to_onehot=True):
		train, test = train_test_split
		self.train_x, self.train_y = train
		self.test_x, self.test_y = test
		if x_flatten:
			self.flatten_x()
		if y_to_onehot:
			self.int_y_arrays_to_onehot()
		for variable in [self.train_x, self.train_y, self.test_x, self.test_y]:
			if not len(variable.shape) == 2:
				raise Exception('Invalid data dimensions')
		self.input_size = self.train_x.shape[1]
		self.output_size = self.train_y.shape[1]
	
	def initialize_parameters(self):
		self.weights, self.biases

		for i in range(self.depth):
			self.weights += [None]
			self.biases += [None]
			self.layer_outputs += [None]

		for prior_layer_idx, prior_layer_size in enumerate(self.architecture):
			curr_layer_idx = prior_layer_idx + 1
			curr_layer_size = self.architecture[curr_layer_idx]

			self.weights[curr_layer_idx] = np.array([
				[self.randomizer('w') for i in range(prior_layer_size)]
					for j in range(curr_layer_size)]

			self.biases[curr_layer_idx] = np.array([
				self.randomizer('b') for in range(curr_layer_size])
			self.biases[curr_layer_idx].reshape(-1, 1)
		
	def __init__(self, data, **kwargs): 
		self.hyperparameters.update(kwargs)
		self.__dict__.update(self.hyperparameters)

		self.sanitize_data(data)

		self.architecture = (
			[self.input_size] + [self.hidden] + [self.output_size])
		self.depth = len(self.architecture)

		self.initialize_parameters()
		self.restricted_gradients = {}
		self.preactivated_outputs = {}
		self.layer_outputs = {}
		self.evaluated = False

	def feedforward(self, X):
		self.layer_outputs[0] = X
		for prior_layer_idx in range(self.depth):
			curr_layer_idx   = prior_layer_idx + 1	
			curr_layer_input = self.layer_outputs[prior_layer_idx]
			layer_function   = self.activated_layer(curr_layer_idx)
			preactivation	 = self.preactivated_layer(curr_layer_idx)

			curr_preactivated = preactivation(curr_layer_input) 
			curr_layer_output = layer_function(curr_layer_input)

			self.layer_outputs[curr_layer_idx] 		  = curr_layer_output
			self.preactivated_outputs[curr_layer_idx] = curr_preactivated

		self.evaluated = True	
		return curr_layer_output
	
	def update(self):
		

	def parameter_gradients(self, idx):
		input_vector 	  		= self.layer_outputs[idx - 1]
		preactivation  	  		= self.preactivated_outputs[idx]	
		incoming_gradient 		= self.restricted_gradients[idx + 1]
		activation 		    	= self.local_activation(idx)
		activation_jacobian     = activation.jacobian(preactivation)
		postactivation_gradient = incoming_gradient @ activation_jacobian
		
		weight_gradient_matrix = np.outer(
			postactivation_gradient, 
			input_vector)

		bias_gradient_vector = postactivation_gradient.reshape(-1,1)
		return weight_gradient_matrix, bias_gradient_vector

	def compute_restricted_gradients(self, X, Y):
		# Computes gradients for all layers with respect to each input
		# while holding weights and biases constant
		cost_function = CostFunction(Y, self.cost)
		if not self.evaluated:
			Y_hat = self.feedforward(X)
		else:
			Y_hat = self.layer_outputs[-1] 
		cost_gradient = cost_function.gradient(Y_hat)
		self.restricted_gradients[self.depth] = cost_gradient

		for layer_idx in reversed(range(self.depth)):
			local_jacobian = self.local_jacobian(
				self.layer_outputs[layer_idx - 1], 
				layer_index)
			prior_gradient = self.restricted_gradients[layer_idx + 1]
			gradient = prior_gradient @ local_jacobian
			self.restricted_gradients[layer_idx] = gradient

	def	preactivated_layer(self, layer_idx):
		weight_function   = LinearMap(self.weights[layer_idx])
		bias_sum_function = FixedSum(self.biases[layer_idx])
		return bias_sum_function.compose(weight_function)
	
	def activated_layer(self, layer_idx):
		activation_function = ActivationFunction(
			self.activation, 
			self.layer_output_dimension[layer_idx])
		preactivation_function = preactivated_layer(layer_idx)
		local_activation = self.local_activation(layer_idx)
		return local_activation.compose(preactivation_function)
	
	def local_activation(self, layer_idx):
		return ActivationFunction(
			self.activation, 
			self.layer_output_dimension[layer_idx])
	
	def local_jacobian(self, input_vector, layer_idx):
		return self.activated_layer(layer_idx).jacobian(input_vector)



