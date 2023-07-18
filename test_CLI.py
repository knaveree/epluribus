import test_NeuralNetork
from test_NeuralNetwork import train, test
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
	@property
	preloads = {
		'mnist' 		: 	lambda : datasets.mnist.load_data(),
		'boston_housing':   lambda : datasets.boston_housing.load_data(),
		'fashion_mnist' : 	lambda : datasets.fashion_mnist.load_data(),
		'imdb' 			: 	lambda : datasets.imdb.load_data(),
		'cifar10' 		: 	lambda : datasets.cifar10.load_data(),
		'cifar100' 		: 	lambda : datasets.cifar100.load_data(),
		'reuters' 		: 	lambda : datasets.reuters.load_data()}

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

	Command line interface commands are formally deliniated in the docstring
	for this class' parse_args method.
	'''

	def __init__(self):
		'''
		Accepts no arguments and initializes the interface. The parser is 
		initialized to accept the following command structure:


		'''
		self.build_parser()
	
	def build_parser(self):
	 """
    This method initializes an ArgumentParser object for the command-line 
	interface. 

    The CLI is structured as follows:

    - `NeuralNetwork`: The main command.
		- `model` : Generate an untrained model.
			- `type` : Currently, `linear` and `neural` models are sup-
			orted. 
			- `--input`: Input dimensions. May be two dimensional, aka 
				`28.28' (as for the MNIST).
			- `--output`: Simple integer to specify output layer size.
			- `--onehot`: Specify true for classifier models.
			- `--hidden` : Hidden layer sizes for neural models, in the 
				following string format: `1000.500.20` using periods as 
				delimiters
			- `--cost` : Will be `meansqr` by default, `meanabs` is also
				supported.
			- `--saveas` : Filepath or name to save pickled model. Default is 
				current directory and `neuralmodel` or `linearmodel`. Custom
				name is recommended to avoid overwriting. 
		- `paraminit` : Generate a custom paraminitializer for weights and
			biases.
			- `--randomization_method` : 
			- `--weight_range` : 
			- `--
        - `train`: A sub-command to train the model.
            - `data`: Filepath to training data.
            - `--model`: Filepath to model. Can be generated from `model`
				command
            - `--epochs`: Specify training epochs; default 10
            - `--learn_rate`: Learning rate; optional; default .05
            - `--param_initializer`: Use a custom parameter initialization
				protocol. 
            - `--output`: An optional output name for the trained network. If 
			not provided, a name is auto-generated.
        - `test`: A sub-command to test the neural network.
            - `data`: The testing data, structured as a pandas DataFrame.

    :return: An ArgumentParser object configured for the CLI.
    :rtype: argparse.ArgumentParser
    """

	def parse_args(self):
		'''
		
		'''
		self.parser.parse_args()
	
	def _repl(self):
		'''

		'''
		pass
	
	def _clinterface(self):
		'''

		'''
		pass

random.seed(10)

def random_vector_onehot(output_size):
	j = random.randint(0, output_size - 1)
	return [1 if i == j else 0 for i in range(10)]

def random_vector_normal(input_size, mu=0, sigma=1):
	return [random.normalvariate(mu, sigma) for i in range(input_size)]

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
def X_data_MNIST(image_height, image_width, dataset_size):
	return np.array([[random_vector_uniform(image_width) 
		for i in range(image_height)]
			for j in range(dataset_size)])

@pytest.fixture
def Y_data_MNIST(output_size, dataset_size):
	return np.array([random_vector_onehot(output_size) 
		for i in range(dataset_size)])

@pytest.fixture
def mock_MNIST(X_data_MNIST, Y_data_MNIST)
	length = X_data.shape[0]
	assert length == Y_data.shape[0]
	train_length = 4 * (length // 5)
	test_length = dataset_size - train_length
	X_train = X_data[:train_length]
	Y_train = Y_data[:train_length]
	X_test = X_data[train_length:]
	Y_test = Y_data[train_length:]
	return (X_train, Y_train), (X_test, Y_test)

def test_preprocess_train(data_path):
	'''Should somehow mock the data without requiring access to the os'''



def test_preprocess_est(data_path):

if __name__ == "__main__":
    interface = ModelInterface()
	interface._parse_args()

