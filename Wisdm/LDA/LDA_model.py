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
	def __init__(self):
		return None
	
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
		if not (load_saved==False) ^ (docs_iterable==False):
			raise Exception('Either train new model or load saved')

		self.docs = list(docs_iterable)
		self.processed_docs = self.docs.map(self.preprocess)
		self.dictionary = gensim.corpora.Dictionary(self.processed_docs)
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

		if bow_or_tfidf in ['tfidf', 'both']:
			self.tfidf = models.TfidfModel(bow_corpus)
			self.corpus_tfidf = tfidf[bow_corpus]

			self.tfidf_model = gensim.models.LdaMulticore(
				self.corpus_tfidf, 
				num_topics=num_topics, 
				id2word=dictionary, 
				passes=passes, 
				workers=workers)

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

	def bucket_sort(self, sortables_series, model_type='tfidf'):
		if not isinstance(sortables_series, pd.Series):
			raise Exception('Input must be valid pandas Series')

		self.sortables_index = sortables_series.index
		self.processed_sortables = sortable_series.map(self.preprocess)
		self.sortables_corpus = [self.dictionary.doc2bow(
			sortable for sortable in self.processed_sortables]

		model = self.tfidf_model if model_type == 'tfidf' else self.bow_model

		self.topic_distribution = [
			model.get_document_topics(bow) for bow in self.sortables_corpus]

		self.sorted = True

	def category_series(self):
		pass	

