#testing 123
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

data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

def lemmatize_stemming(text):
	stemmer = SnowballStemmer('english', ignore_stopwords=False)
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if (token not in gensim.parsing.preprocessing.STOPWORDS 
			and len(token) > 3):
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 4310].values[0][0]

#processed_docs = documents['headline_text'].map(preprocess)
directory = '/Users/nathanavery/epluribus/Wisdm/LDA'
pd_filename = 'processed_docs.pkl' 
with open(os.path.join(directory, pd_filename), 'rb') as file:
	processed_docs = pickle.load(file)

#dictionary = gensim.corpora.Dictionary(processed_docs)
#dictionary.filter_extremes(
#	no_below=15, 
#	no_above=0.5, 
#	keep_n=100000)

dic_filename = 'dictionary.pkl' 
#with open(os.path.join(directory, dic_filename), 'wb') as file:
#	pickle.dump(dictionary, file)

with open(os.path.join(directory, dic_filename), 'rb') as file:
	dictionary = pickle.load(file)

#LDA using bag of words approach
#bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#with open(os.path.join(directory, 'bow_corpus.pkl'), 'wb') as file:
#	pickle.dump(bow_corpus, file)

with open(os.path.join(directory, 'bow_corpus.pkl'), 'rb') as file:
	bow_corpus = pickle.load(file)

bow_doc_4310 = bow_corpus[4310]

#lda_model = gensim.models.LdaMulticore(
#	bow_corpus, 
#	num_topics=10, 
#	id2word=dictionary, 
#	passes=2, 
#	workers=2)

#with open(os.path.join(directory, 'lda_model.pkl'), 'wb') as file:
#	pickle.dump(lda_model, file)

with open(os.path.join(directory, 'lda_model.pkl'), 'rb') as file:
	lda_model = pickle.load(file)

#LDA using TF-IDF (vectorization)
#tfidf = models.TfidfModel(bow_corpus)

#with open(os.path.join(directory, 'tfidf.pkl'), 'wb') as file:
#	pickle.dump(tfidf, file)

with open(os.path.join(directory, 'tfidf.pkl'), 'rb') as file:
	tfidf = pickle.load(file)

corpus_tfidf = tfidf[bow_corpus]

#lda_model_tfidf = gensim.models.LdaMulticore(
#	corpus_tfidf, 
#	num_topics=10, 
#	id2word=dictionary, 
#	passes=2, 
#	workers=4)

#with open(os.path.join(directory, 'lda_model_tfidf.pkl'), 'wb') as file:
#	lda_model_tfidf = pickle.dump(lda_model_tfidf, file)

with open(os.path.join(directory, 'lda_model_tfidf.pkl'), 'rb') as file:
	lda_model_tfidf = pickle.load(file)

pdb.set_trace()
