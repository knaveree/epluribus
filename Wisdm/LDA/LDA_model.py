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

class LDAModel:
	def __init__(self, docs_iterable):
		self.tfidf_trained = False
		self.bow_trained = False
		self.sortables_are_categorized = False

		if not isinstance(docs_iterable, pd.Series):
			try: 
				self.docs = pd.Series(data = list(docs_iterable))
			except: 
				raise Exception('Invalid documents format')
		else:
			self.docs = docs_iterable

		self.processed_docs = 	self.autoload( 
			lambda : self.docs.map(self.preprocess),
			'processed_docs.pkl')

		self.dictionary = 		self.autoload(
			lambda : gensim.corpora.Dictionary(self.processed_docs),
			'dictionary.pkl')

	def autoload(self, obj_generator, filename, rewrite=False, regenerate=False):
		filepath = os.path.join(os.getcwd(), filename)
		found = False

		try:
			with open(filepath, 'rb') as file:
				obj = pickle.load(filepath)
			found = True
		except:
			pass	
		
		if found and not regenerate:
			return obj

		elif (not found) or rewrite:
			obj = obj_generator()
			with open(filepath, 'wb') as file:
				pickle.dump(obj, file)
				file.close()
			return obj
				
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
				result.append(self.lemmatize_stemming(token))
		return result
	
	def train(
		self, 
		retrain = False,
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

		self.bow_corpus = [self.dictionary.doc2bow(doc) 
			for doc in self.processed_docs]

		if method in ['bow', 'both']:
			bow_generator = lambda : gensim.models.LdaMulticore(
				self.bow_corpus, 
				num_topics=10, 
				id2word=self.dictionary, 
				passes=passes, 
				workers=workers)
			self.bow_model = self.autoload(
				bow_generator,
				'bow_model.pkl',
				rewrite=retrain)

			self.bow_trained = True
			
		if method in ['tfidf', 'both']:
			self.tfidf = models.TfidfModel(bow_corpus)
			self.corpus_tfidf = tfidf[bow_corpus]

			tfidf_generator = lambda : gensim.models.LdaMulticore(
				self.corpus_tfidf, 
				num_topics=num_topics, 
				id2word=self.dictionary, 
				passes=passes, 
				workers=workers)
			self.tfidf_model = self.autoload(
				tfidf_generator,
				'tfidf_model.pkl',
				rewrite=retrain)

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

		processed_sortables = sortables_series.map(self.preprocess)
		tokenized_sortables = [
			self.dictionary.doc2bow(sortable) 
				for sortable in processed_sortables]
		tokenized_series = pd.Series(
			data = tokenized_sortables,
			index = sortables_series.index)

		distributions = tokenized_series.map(model.get_document_topics)

		def get_dominant_topic(distribution):
			probability = lambda topic : topic[1]
			dominant_topic = max(distribution, key = probability)[0]
			return dominant_topic 

		self.categorized_sortables_series = distributions.map(
			get_dominant_topic)
		self.sortables_are_categorized = True	

	def category_series(self):
		'''Outputs a Pandas Series with 1 to 1 category assignments assigned
		to the sample index'''
		if not self.sortables_are_categorized:
			raise Exception('First input sortable pd.Series data')
		return self.categorized_sortables_series

