import test_NeuralNetork
from test_NeuralNetwork import train, test
import argparse
import random
import pandas as pd
import pickle

def load(saved_model):
	pass

def unpack(csv_file):
	pass

def main():
	'''
	===============================================
	Command Line Interface for NeuralNetwork Module
	===============================================

	This is a simple interface for the NeuralNetwork Module. It can be mani-
	pulated directly from the command line using the command line arguments
	enumerated below. It can also be prompted to activate a REPL that prov-
	ides more explicit and intuitive navigation of the module's features.

	Command Line Arguments
	======================
		
	
	'''
def repl():
	pass

def preprocess_data(data_path : str) -> tuple:
	'''Takes in filepath to a csv_file and outputs X and Y tuple as lists of 
	lists.'''
	pass

random.seed(10)

def random_y(output_size):
	j = random.randint(0, output_size - 1)
	return [1 if i == j else 0 for i in range(10)]

def random_x(input_size, mu=0, sigma=1):
	return [random.normalvariate(mu, sigma) for i in range(input_size)]

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
def X_data(image_height, image_width, dataset_size):
	return np.array([[random_x(image_width) 
		for i in range(image_height)]
			for j in range(dataset_size)])

@pytest.fixture
def Y_data(output_size, dataset_size):
	return np.array([random_y(output_size) 
		for i in range(dataset_size)])

@pytest.fixture
def train_test_split(X_data, Y_data)
	length = X_data.shape[0]
	assert length == Y_data.shape[0]
	train_length = 4 * (length // 5)
	test_length = dataset_size - train_length
	X_train = X_data[:train_length]
	Y_train = Y_data[:train_length]
	X_test = X_data[train_length:]
	Y_test = Y_data[train_length:]
	return (X_train, Y_train), (X_test, Y_test)
		
def preprocess_data(train_test_split):
	train_data, test_data = train_test_split
	
def test_preprocess_data(data_path)
	'''Should somehow mock the data without requiring access to the os'''

#if __name__ == "__main__":
#    main()

