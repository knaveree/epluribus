class ScalarFunction:
	evaluators = {
		'ReLu' : lambda x : x if x > 0 else 0,
		'Logistic' : 'Unsupported', 
		'MeanSquared' : lambda Y_hat, Y : 

	}
	primes = {
		'ReLu' : lambda x : 1 if x > 0 else 0,
		'Logistic' : 'Unsupported',

	}
	def vectorize(self):
		return VectorFunction(self)

	def __init__(self, name):
		self.name = name

class VectorFunction:
	


class NeuralNetwork:
	def __init__(
		self,
		keras_type_data : tuple,
		hidden_layers : list,
		param_initializer : ParamInitializer = ParamInitializer(),
		activation_function : ('Relu', 'Logistic') = 'ReLu',
		cost_function : ('MeanAbsolute, MeanSquared') = 'MeanSquared',
		output_type : ('OneHot', 'Identity') = 'OneHot')



class NeuralPublic:
