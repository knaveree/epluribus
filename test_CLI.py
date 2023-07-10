from test_NeuralNetwork import train, test
import argparse
import random
import pandas as pd

def unpickle(saved_model):
	pass

def unpack(csv_file):
	pass

def main():
	parser = argparse.ArgumentParser(
		description = 'Command line interface for Neural Network Module')
	subparsers = parser.add_subparsers(dest='command')

	#################################################
	parser_train = subparsers.add_parser(
		'train',
		help='Train the model')
	parser_train.add_argument(
		'--data',
		type=str,
		required=True,
		help='Filepath to training data csv file')
	parser_train.add_argument(
		'--model',
		dest='model_path',
		type=str
		help='Filepath to existing model to retrain.')
	parser_train.add_argument(
		'--saveas',
		dest='save_path',
		type=str,
		default='./NN_current.bin')
		help='Filepath to save trained model')
	parser_train.add_argument(
		'--epochs',
		help='Number of epochs to use for training cycle')
		type=int,
		default=10)
	parser_train.add_argument(
		'--learn_rate',
		dest='learn_rate'
		type=float,
		default=.05,
		help='Learn rate to use for backpropagation')

	##################################################
	parser_test=subparsers.add_parser(
		'test',
		help='Test the model')
	parser_test.add_argument(
		'--model',
		dest='model_path',
		type=str
		help='Filepath to model to be tested.')
	parser_test.add_argument(
		'--data',
		dest='data',
		type=str,
		required=True,
		help='Filepath to testing data')
	
	parser.parse_args()

	if parser.model_path:
		pass
	if parser.command == 'train':
		pass
	elif parser.command == 'test':
		pass

if __name__ == "__main__":
    main()

