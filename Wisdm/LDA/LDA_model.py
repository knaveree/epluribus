import pandas as pd
import pdb
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
import numpy as np
import nltk
import pickle
import os
nltk.download('wordnet')
np.random.seed(2018)

class LDAModel:
	def __init__(self, docs_iterable):
		self.tfidf_trained = False
		self.bow_trained = False
		self.sortables_are_categorized = False

		if not isinstance(self.docs, pd.Series):
			try: 
				self.docs = pd.Series(data = list(docs_iterable))
			except: 
				raise Exception('Invalid documents format')
		else:
			self.docs = docs_iterable

		self.processed_docs = self.docs.map(self.preprocess)
		self.dictionary = gensim.corpora.Dictionary(self.processed_docs)

	def load_kernel(self, filepath, model_type=False):
		if not model_type in ('bow', 'tfidf'):
			raise Exception('Choose bow or tfidf designation for model')
			
		with open(os.path.join(os.path.cwd(), filepath), 'rb') as file:
			model = pickle.load(file)

		if model_type == 'bow':
			self.bow_trained = True
			self.bow_model = model

		if model_type == 'tfidf':
			self.tfidf_trained = True
			self.tfidf_model = model

	def is_trained(self):
		return self.bow_trained or self.tfidf_trained

	def lemmatize_stemming(self, text):
		stemmer = SnowballStemmer('english', ignore_stopwords=False)
		return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

	def preprocess(self, text):
		result = []
		for token in gensim.utils.simple_preprocess(text):
			if (token not in gensim.parsing.preprocessing.STOPWORDS 
				and len(token) > 3):
				result.append(lemmatize_stemming(token))
		return result
	
	def train(
		self, 
		docs_iterable, 
		num_topics = 10,
		method='bow',
		no_below = 15,
		no_above = 0.5,
		keep_n = 100000,
		passes=2,
		workers=2):
	
		if not method in ['bow', 'tfidf', 'both']:
			raise Exception('Choose bow tfidf or both')

		self.dictionary.filter_extremes(
			no_below=no_below, 
			no_above=no_above, 
			keep_n=keep_n)

		self.bow_corpus = [dictionary.doc2bow(doc) 
			for doc in self.processed_docs]

		if bow_or_tfidf in ['bow', 'both']:
			self.bow_model = gensim.models.LdaMulticore(
				self.bow_corpus, 
				num_topics=10, 
				id2word=self.dictionary, 
				passes=passes, 
				workers=workers)

			self.bow_trained = True

		if bow_or_tfidf in ['tfidf', 'both']:
			self.tfidf = models.TfidfModel(bow_corpus)
			self.corpus_tfidf = tfidf[bow_corpus]

			self.tfidf_model = gensim.models.LdaMulticore(
				self.corpus_tfidf, 
				num_topics=num_topics, 
				id2word=dictionary, 
				passes=passes, 
				workers=workers)

			self.tfidf_trained = True

	def bucket_sort(self, sortables_series, model_type='bow'):
		if not isinstance(sortables_series, pd.Series):
			raise Exception('Input must be valid pandas Series')

		if model_type == 'tfidf': 
			if not self.tfidf_trained:
				raise Exception('Model not yet trained via tfidf')
			model = self.tfidf_model
		elif model_type == 'bow':
			if not self.bow_trained:
				raise Exception('Model not yet trained via bow')
			model = self.bow_model 
		else:	
			raise Exception('Model type must be either bow or tfidf')

		processed_sortables = sortable_series.map(self.preprocess)
		tokenized_sortables = [
			self.dictionary.doc2bow(sortable) 
				for sortable in processed_sortables]
		tokenized_series = pd.Series(
			data = tokenized_sortables,
			index = sortables_series.index)

		distributions = tokenized_series.map(model.get_document_topics)

		def get_dominant_topic(distribution)
			probability = lambda topic : topic[1]
			most_probable_topic = max(distribution, key = probability)
			identifier = most_probable_topic[0]
			return identifier

		self.categorized_sortables_series = distributions.map(
			get_dominant_topic)
		self.sortables_are_categorized = True	

	def category_series(self):
		'''Outputs a Pandas Series with 1 to 1 category assignments assigned
		to the sample index'''
		if not self.sortables_are_categorized:
			raise Exception('First input sortable pd.Series data')
		return self.categorized_sortables_series

