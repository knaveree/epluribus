import test_NeuralNetork
from test_NeuralNetwork import *
import argparse
import random
import pandas as pd
import pickle
from unittest.mock import patch
from tensorflow.keras import datasets

def load(saved_model):
	pass

def unpack(csv_file):
	pass

class ModelInterface:
	'''
	=====================
	Model Interface Class
	=====================
	This is a simple interface for the NeuralNetwork Module. It can be mani-
	pulated directly from the command line using the command line arguments
	enumerated below. It can also be prompted to activate a REPL that prov-
	ides more explicit and intuitive navigation of the module's features.

	Data fed to the system is assumed to be properly preprocessed and avail-
	able to the user in CSV form. Models are assumed to be pickled Python
	objects. Data will be automatically split at an 80/20 ratio during con-
	version from CSV. 

	Finally, the following Keras datasets are directly available to the user
	by using their names in lieu of a local filepath for the 'datapath' kwarg:
	
	'mnist'
	'boston_housing'
	'imdb'
	'reuters'
	'fashion_mnist'
	'cifar10'
	'cifar100'

	These datasets can also be accessed directly as properties of the Model-
	Interface class through the syntax ModelInterface.preloads['<dataset>']()

	Command line interface commands are formally delineated in the docstring
	for this class' build_parser method.
	'''

	def __init__(self):
		self.parser = self.build_parser()
	
	def parse_list(self, arg):
		arg = re.sub(r'[\s\(\)]', '', arg)
		arg = re.sub(r'[.;]', ',', arg)
		parts = arg.split(',')
		try:
			return [float(part) for part in parts]
		except ValueError:
			raise argparse.ArgumentTypeError(f"Invalid number value: {arg}")

	def join_path(self, filename, directory='./'):
		if os.path.isabs(filename):
			return filename
		else:
			return os.path.join(directory, filename)

	def build_parser(self):
		parser = argparse.ArgumentParser()
		subparsers = parser.add_subparsers(dest='command')
		
		#subparser for generate_model command
		#####################################
		parser_generate_model = subparsers.add_parser(
			'generate_model',
			help='Generate an untrained model.')
		parser_generate_model.add_argument(
			'type', 
			choices=['linear', 'neural'],
			default='neural',
			help='Currently, `linear` and `neural` models are supported.')
		parser_generate_model.add_argument(
			'--data',
			'-d',
			required=True,
			type=str,
			help='Filepath to data or name of built in Keras dataset')
		parser_generate_model.add_argument(
			'--normalize',
			'-n',
			default=True,
			type=bool,
			help='Indicate whether or not to normalize X data in range (0,1)')
		parser_generate_model.add_argument(
			'--classify',
			'-c',
			default=True,
			type=bool,
			help='Indicate whether or not to convert Y to onehot encoding')
		parser_generate_model.add_argument(
			'--format',
			'-f',
			choices=['csv','native'],
			default='native',
			required=True,
			type=str,
			help='Specify if data is in csv or keras-native format.')
		parser_generate_model.add_argument(
			'-e',
			'--evaluation', 
			choices=['onehot','identity'],
			default='onehot',
			help='Optional; `onehot` for classifier models and `identity` to 
			leave output unchanged. Default `onehot`.')
		parser_generate_model.add_argument(
			'-a',
			'--activation',
			choices=['ReLu', 'Logistic'],
			default=False,
			help='Optional; activation function for neural models. Options
			are `ReLu`, `Logisitic` and  Defaults to `ReLu`.')
		parser_generate_model.add_argument(
			'-h',
			'--hidden',
			type=self.parse_list,
			required=True,
			help='Hidden layer sizes for neural models, in the following 
			string format: `1000.500.20` using periods, commas or semicolons
			as delimiters')
		parser_generate_model.add_argument(
			'-c',
			'--cost',
			choices=['MeanSquared','MeanAbsolute'],
			default=CostFunction('MeanSquared'),
			required=False,
			help='Optional; specifies cost function, will be `meansqr` by 
			default, `meanabs` is also supported.')
		parser_generate_model.add_argument(
			'-p',
			'--param_initializer', 
			default=False,
			help='Optional; Use a custom parameter initialization protocol 
			by providing filepath to pickled initializer. (See 
			param_initializer help string for more information on generating 
			these.) Else, default values for the class will be used.')
		parser_generate_model.add_argument(
			'-s',
			'--saveas',
			default='./model.pkl',
			required=False,
			help='Optional filepath to save pickled model. Default is 
			\'./model.pkl\'. Custom name can be used to avoid overwriting.') 

		#subparser for param_initializer command
		#####################################
		parser_param_initializer = subparsers.add_parser(
			'param_initializer',
			help='Generate custom parameter initializer. This allows for 
			defining the distributions from which initial weights and biases 
			are to be randomly generated.')
		parser_param_initializer.add_argument(
			'saveas',
			default='./initializer.pkl',
			required=False,
			help='Filepath to save custom initializer. Must be in same directory
			as model. Defaults to \'./initializer.pkl\'')
		parser_param_initializer.add_argument(
			'-m',
			'--method',
			default='normal',
			choices=['normal','xavier','uniform'],
			help='Optional; default `normal`; `xavier` and `uniform` also 
			supported. Note: `xavier` requires specifying the subcommand 
			`--input_layer_size` or you\'ll get an exception thrown at you.')
		parser_param_initializer.add_argument(
			'-w',
			'--weight_range',
			required=False,
			type=self.parse_list,
			help='Optional; default [-.3, 3]. If the randomization method is
			uniform, these are the bounds. For other distributions, this
			represents plus/minus one standard deviation around the mean.
			Should be entered as a tuple of integers using parentheses')
		parser_param_initializer.add_argument(
			'-b',
			'--bias_range',
			type=self.parse_list,
			required=False,
			help='Optional; default [0, 0]. If the randomization method is
			uniform, these are the bounds. For normal distribution, this
			represents plus/minus one standard deviation around the mean.')
		parser_param_initializer.add_argument(
			'-i',
			'--input_layer_size',
			type=int,
			help='Optional; only needed if using xavier for randomization 
			method; define an input layer size with an integer'

		#subparser for train command
		#####################################
		parser_train = subparsers.add_parser('train')
		parser_train.add_argument(
			'--model',
			'-m',
			required=True,
			help='Filepath to pickled model. Model can be initialized using 
			generate_model command.')
		parser_train.add_argument(
			'--data',
			'-d',
			required=False,
			default=False,
			help='Filepath to novel training data. Can also use the 
			predefined keras dataset names here, without need for a filepath
			This is for data beyond what model was originally initialized 
			with.')
		parser_train.add_argument(
			'--normalize',
			'-n',
			default=True,
			type=bool,
			help='Indicate whether or not to normalize X data in range (0,1)')
		parser_train.add_argument(
			'--classify',
			'-c',
			default=True,
			type=bool,
			help='Indicate whether or not to convert Y to onehot encoding')
		parser_train.add_argument(
			'--epochs',
			'-e',
			default=10,
			type=int,
			help='Optional; default 10; specify training epochs.')
		parser_train.add_argument(
			'--learn_rate',
			'-lr',
			type=float,
			default=.05,
			help='Optional; learning rate for gradient descent. Default is .05')
		parser_train.add_argument(
			'--saveas',
			'-s',
			required=True,
			help='Required; a new filepath for the trained network checkpoint.')
		parser_train.add_argument(
			'--display',
			'-d',
			type=bool,
			default=True,
			help='Optional; choose whether or not to display the cumulative 
			run-loss as a function of total steps after the training cycle 
			concludes. Defaults to true.')

		#subparser for test command
		#####################################
		parser_test = subparsers.add_parser('test')
		parser_train.add_argument(
			'--model',
			'-m',
			required=True,
			help='Filepath to pickled model. Model can be initialized using 
			generate_model command.')
		parser_test.add_argument(
			'--data',
			default=False,
			help='Any novel testing data, structured in Keras format. Any
			Keras dataset can be used via only its name and no need
			for a filepath. A limitation of this format is that you must
			structure data as a train test split for all data, even if 
			there\'s nothing in the train part.')
		parser_test.add_argument(
			'--normalize',
			'-n',
			default=True,
			type=bool,
			help='Indicate whether or not to normalize X data in range (0,1)')
		parser_test.add_argument(
			'--classify',
			'-c',
			default=True,
			type=bool,
			help='Indicate whether or not to convert Y to onehot encoding')
		parser_test.add_argument(
			'--display', 
			'-d',
			required=False,
			help='Optional; choose whether or not to display the cumulative 
			accuracy as a function of total data points fed to the model.')
	
	def _repl(self):
		'''

		'''
		pass
	
	def _load_data(self, model, args):
		a = args
		raw_data = self._load(a.data)
		model._preprocess(raw_data, a.normalize, a.classify)
	
	def _load(self, filepath):
		filepath = self.join_path(filepath)
		with open(filepath, 'rb') as file:
			obj = pickle.load(file)	
		return obj	
	
	def _dump(self, obj, filepath):
		with open(filepath, 'wb') as file:
			pickle.dump(obj, file)
	
	def interface(self):
		 """
		The CLI is structured as follows:

		- `generate_model` : Generate an untrained model.
			- `type` : Currently, `linear` and `neural` models are sup-
				orted. 
			- `--data` : Filepath to pickled data or name of Keras dataset.
			- `--normalize` : Defaults to True; will normalize data between
				1 and 0.
			- `--classify` : Optional; defaults to True; turns integer y
				values into onehot vectors.
			- `--evaluation`: Specify `onehot` for classifier models
				and `identity` to leave output unchanged. Default `onehot`.
			- `--activation` : Activation function for neural models. Options
				are `ReLu` and `Logisitic.' Defaults to `ReLu`.
			- `--hidden` : Hidden layer sizes for neural models, in the 
				following string format: `1000.500.20` using periods as 
				delimiters
			- `--cost` : Will be `meansqr` by default, `meanabs` is also
				supported.
			- `--param_initializer`: Use a custom parameter initialization
				protocol. Else, default values for the class will be used. 
			- `--saveas` : Filepath or name to save pickled model. Default is 
				current directory and `neuralmodel` or `linearmodel`. Custom
				name can be used to avoid overwriting, but not required given
				that untrained models are not themselves that valuable. 

		- `param_initializer` : Generate a custom initializer for weights 
			and biases.
			- `saveas` : Filepath for custom initializer. Defaults to
			./initializer.pkl
			- `--method` `-m` : Optional; default `normal`; `xavier` 
				and `uniform` also supported. 
			- `--weight_range` `-w` : Optional; default (-.3,.3)
			- `--bias_range` `-b`: Optional; default (0,0)
			- `--input_layer_size` `-i`: Optional; only needed if using Xavier 
			for randomization method; define an input layer size with an 
			integer

		- `layer_history`: A command for visualizing gradient magnitudes
			across all steps for a given layer in a given training cycle. 
			This can help to identify vanishing and exploding gradients. 
				- `model` : Filepath to trained model. Model must be trained.
				- `--layer` `-l`: Required; choose layer; first layer is 0. 
				Layers can also be reverse indexed; -1 for last layer.

		- `step_history` : For visualizing gradient magnitudes across the 
			entire network in a given step. This can help to identify 
			vanishing and exploding gradients. 
				- `model` : Filepath to trained model. Model must be trained.
				- `--epoch` `-e`: Required; choose epoch. 
				- `--step` `s`: Optional; choose step within epoch. Defaults
				to final step in training epoch. 

		- `train`: A sub-command to train the model via gradient descent.
			- `--model` `-m`: Filepath to model. Model can be initialized 
				using generate_model command. 
			- `--data` `-d`: Filepath to training data. Can also use the 
				predefined keras dataset names here. Data must be in Keras 
				format. 
			- `--epochs` `-e`: Optional; specify training epochs; default 10.
			- `--learn_rate` `-lr`: Optional; learning rate for gradient 
				descent. Default is .05.
			- `--saveas` `-s`: Required; a new name for the trained network
				checkpoint.
			- `--display` `-d`: Optional; choose whether or not to display 
				the cumulative run-loss as a function of total steps after
				the training cycle concludes. Defaults to True. 

		- `test`: A sub-command to test an existing model.
			- `data`: The testing data, structured as a pandas DataFrame. Data
				must be in Keras format. 
			- `--display` `-d`: Optional; choose whether or not to display 
				the cumulative accuracy as a function of total data points
				fed to the model.

		:return: An ArgumentParser object configured for the CLI.
		:rtype: argparse.ArgumentParser
		"""
		args = self.parser.parse_args()
		a = args
		save_obj = False
		if args.command == 'generate_model':
			if args.type == 'neural':
				save_obj = NeuralPublic(args)	
			elif args.type == 'linear':
				save_obj = LinearPublic(args)
		elif args.command == 'param_initializer':
			save_obj = ParamPublic(args)
		elif args.command == 'layer_history':
			model = self._load(model)	
			model.display_layer_history(a.layer)
		elif args.command == 'step_history':
			model = self._load(model)	
			model.display_step_history(a.epoch, a.step)
		elif args.command == 'train':
			model = self._load(a.model)
			if a.data:	
				self._load_data(model, args)
			model.train(
				epochs = a.epochs,
				learn_rate = a.learn_rate,
				display = a.display)
			save_obj = model
		elif args.command == 'test':
			model = self._load(a.model)
			if a.data:	
				self._load_data(model, args)
			model.test(display=a.display)
		if not save_obj == False:
			self._dump(save_obj, a.saveas)

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
def interface_calls():
	return {
		'generate_model' : 'generate_model mnist '
	}

		 """
		- `generate_model` : Generate an untrained model.
			- `type` : Currently, `linear` and `neural` models are sup-
				orted. 
			- `--data` : Filepath to pickled data or name of Keras dataset.
			- `--normalize` : Defaults to True; will normalize data between
				1 and 0.
			- `--classify` : Optional; defaults to True; turns integer y
				values into onehot vectors.
			- `--evaluation`: Specify `onehot` for classifier models
				and `identity` to leave output unchanged. Default `onehot`.
			- `--activation` : Activation function for neural models. Options
				are `ReLu` and `Logisitic.' Defaults to `ReLu`.
			- `--hidden` : Hidden layer sizes for neural models, in the 
				following string format: `1000.500.20` using periods as 
				delimiters
			- `--cost` : Will be `meansqr` by default, `meanabs` is also
				supported.
			- `--param_initializer`: Use a custom parameter initialization
				protocol. Else, default values for the class will be used. 
			- `--saveas` : Filepath or name to save pickled model. Default is 
				current directory and `neuralmodel` or `linearmodel`. Custom
				name can be used to avoid overwriting, but not required given
				that untrained models are not themselves that valuable. 

		- `param_initializer` : Generate a custom initializer for weights 
			and biases.
			- `saveas` : Filepath for custom initializer. Defaults to
			./initializer.pkl
			- `--method` `-m` : Optional; default `normal`; `xavier` 
				and `uniform` also supported. 
			- `--weight_range` `-w` : Optional; default (-.3,.3)
			- `--bias_range` `-b`: Optional; default (0,0)
			- `--input_layer_size` `-i`: Optional; only needed if using Xavier 
			for randomization method; define an input layer size with an 
			integer

		- `layer_history`: A command for visualizing gradient magnitudes
			across all steps for a given layer in a given training cycle. 
			This can help to identify vanishing and exploding gradients. 
				- `model` : Filepath to trained model. Model must be trained.
				- `--layer` `-l`: Required; choose layer; first layer is 0. 
				Layers can also be reverse indexed; -1 for last layer.

		- `step_history` : For visualizing gradient magnitudes across the 
			entire network in a given step. This can help to identify 
			vanishing and exploding gradients. 
				- `model` : Filepath to trained model. Model must be trained.
				- `--epoch` `-e`: Required; choose epoch. 
				- `--step` `s`: Optional; choose step within epoch. Defaults
				to final step in training epoch. 

		- `train`: A sub-command to train the model via gradient descent.
			- `--model` `-m`: Filepath to model. Model can be initialized 
				using generate_model command. 
			- `--data` `-d`: Filepath to training data. Can also use the 
				predefined keras dataset names here. Data must be in Keras 
				format. 
			- `--epochs` `-e`: Optional; specify training epochs; default 10.
			- `--learn_rate` `-lr`: Optional; learning rate for gradient 
				descent. Default is .05.
			- `--saveas` `-s`: Required; a new name for the trained network
				checkpoint.
			- `--display` `-d`: Optional; choose whether or not to display 
				the cumulative run-loss as a function of total steps after
				the training cycle concludes. Defaults to True. 

		- `test`: A sub-command to test an existing model.
			- `data`: The testing data, structured as a pandas DataFrame. Data
				must be in Keras format. 
			- `--display` `-d`: Optional; choose whether or not to display 
				the cumulative accuracy as a function of total data points
				fed to the model.

		:return: An ArgumentParser object configured for the CLI.
		:rtype: argparse.ArgumentParser
		"""
@pytest.fixture
def parsers:
