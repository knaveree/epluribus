import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from copy import copy
import pdb
import datetime as dt

class TestingSentinel:
	def foo(self, bar):
		pass

	#Simulates some approximation of API response from database in 
	#the form of a pandas dataframe. The response represents however many
	#days worth of headlines are requested, based on a date range via
	#either negative integers counting back from -1 (following reverse
	#indexing convention) or two python datetime objects.

	def __init__(self):
		source = 'abcnews-date-text.csv'
		self.df = pd.read_csv(source)
		self.end_date_string = '20211231'
		self.end_date_dto = dt.datetime(year=2021, month=12, day=31)
	
	def format_request(self, pre_request):
		dynamic = {
			'batch_id' : lambda row : self.int_date_as_ordinal(row['date']),
			'record_id' :lambda row : hash(tuple(row['title']))
		}
		simulated = {
			'content' 	: lambda row: 'There once was a man from Nantucket...',
			'author' 	: lambda row: 'Ernest Hemingway',
			'timestamp' : lambda row: dt.datetime.now(),
			'url' 		: lambda row: 'http://wisdm.news',
			'source' 	: lambda row: 'ABC News',
		}
		novel = {
			'description' 	: lambda row : None,
			'bucket_id' 	: lambda row : None,
		}
		renames = {
			'publish_date' : 'date',
			'headline_text' : 'title'
		}
		
		pre_request = pre_request.rename(columns=renames)
		for functions_dict in [dynamic, simulated, novel]:
			for column_name, func in functions_dict.items():
				pre_request[column_name] = pre_request.apply(func, axis=1)
		request = pre_request.set_index('record_id')
		return request	

	def int_date_as_ordinal(self, n):
		return self.int_as_dto(n).toordinal()	

	def strng_as_dto(self, strng):
		year, month, day = int(strng[0:4]), int(strng[4:6]), int(strng[6:8])
		return dt.datetime(year=year, month=month, day=day)
	
	def int_as_dto(self, n):
		return self.strng_as_dto(str(n))

	def dto_as_strng(self, dto):
		return str(self.dto_as_int(dto))
	
	def dto_as_int(self, dto):
		date_int = dto.year * 10000 + dto.month * 100 + dto.day
		return date_int

	def req_by_dto(self, start_dto, end_dto=None):
		if end_dto==None:
			end_dto=self.end_date_dto
		pre_request = self.df[
			(self.df['publish_date'] >= self.dto_as_int(start_dto)) &
			(self.df['publish_date'] <= self.dto_as_int(end_dto))]
		return self.format_request(pre_request)

	def req_by_days(self, start_days, end_days=-1):
		end_dto = self.end_date_dto + dt.timedelta(days=1+end_days)
		start_dto = self.end_date_dto + dt.timedelta(days=1+start_days)
		return self.req_by_dto(start_dto, end_dto=end_dto)

class BucketSorter:
	def __init__(self, formatted_dataframe):
		self.vectorizer = TfidfVectorizer()

		self.raw_data = formatted_dataframe 

		func = self._preprocess_each_title
		processed_titles = self.raw_data['title'].apply(func)
		self.matrix = self.vectorizer.fit_transform(processed_titles)

		self.decomposed = 	False 
		self.pruned 	=   False
		self.sorted 	= 	False

	def _preprocess_each_title(self, title):
		words = word_tokenize(title.lower())
		words = [w for w in words if w.isalpha() and w not in stop_words]
		return ' '.join(words)			

	def normalize(self):
		column_means = np.mean(self.matrix, axis=0)	
		self.data_matrix = self.matrix - column_means

	def svd(self, n=100, target_variance=.95, override=False, normalize=False): 
		t = target_variance
		self.truncated_svd = TruncatedSVD(n_components=n)

		if normalize:
			self.normalize()
			normalized = True
			matrix = self.data_matrix
		else:
			matrix = self.matrix

		matrix_reduced = self.truncated_svd.fit_transform(matrix)
		v = self.truncated_svd.explained_variance_ratio_.sum()
		if v < t:
			if override==True:
				print(f'Proceeding with variance {v} < target {t}')
			else:
				raise Exception(f'Variance {v} < target {t} for n={n}')
		else:
			print(f'Variance {v} >= target {t} for n={n}')	
		self.df_svd_reduced = pd.DataFrame(
			matrix_reduced, 
			index=self.raw_data.index)
		self.decomposed = True

	def k_means(self, num_clusters):
		if self.decomposed == False:
			raise Exception('Run svd method prior to k-means')
		kmeans = KMeans(
			n_clusters=num_clusters, 
			init='k-means++', 
			max_iter=1000, 
			random_state=42)
		kmeans.fit(self.matrix_reduced)
		self.xor_bucket_series = pd.Series(
			kmeans.labels_, 
			index=self.df_svd_reduced.index)

	def prune_records(self, euclidean_threshold=.9):
		if self.decomposed == False:
			raise Exception('Run BucketSorter().svd method before pruning')

		def passes_threshold(row):
			return np.sum((row.values)**2) >= euclidean_threshold

		pruning_mask = self.df_svd_reduced.apply(passes_threshold, axis = 1)
		self.inverted_mask = ~pruning_mask

		self.df_svd_reduced = self.df_svd_reduced[pruning_mask]
		self.matrix_reduced = self.df_svd_reduced.values
		self.pruned = True

	def sort(
		self, 
		num_clusters		=10, 
		svd_n				=200, 
		euclidean_threshold	=.9,
		svd_target_var		=.5,
		override_svd		=True,
		normalize			=True):

		self.svd(
			n=svd_n, 
			target_variance=svd_target_var, 
			override=override_svd,
			normalize=normalize)
		if euclidean_threshold:	
			self.prune_records(euclidean_threshold=euclidean_threshold)
		self.k_means(num_clusters)
		self.sorted = True

	def iterative_sort(self, **kwargs):
		default_kwargs = {
			'cycles'				:	 3,      	
			'num_clusters'			:	 [5, 5, 5],
			'svd_n'					:	 200, 
			'euclidean_threshold'	:	 [.9, .75, .5],
			'svd_target_var'		:	 .5,
			'override_svd'			:	 True,
			'normalize'				:	 True}

		default_kwargs.update(kwargs)
		cycles = default_kwargs['cycles']

		for varname, value in default_kwargs.items():
			if varname == 'cycles':
				assert isinstance(value, int)
			else:
				try: 
					iterable = iter(value)
					length = sum(1 for x in iterable)
					if not length == cycles:
						raise Exception('Malformed iterable argument')
				except:
					default_kwargs[varname] = [value for i in range(cycles)]
		iterative_sort_kwargs = default_kwargs

		iterative_sorter = IterativeSorter(copy(self.raw_data))
		iterative_sorter.iterative_sort(**iterative_sort_kwargs)
		self.xor_bucket_series = iterative_sorter.xor_bucket_series

		self.sorted=True
				
	def extract(self, as_series=False, as_dataframe=False):
		'''as_series=True returns a pandas Series with the Xor bucket values 
		matched to each article_id, and includes null values for unassigned 
		articles. This is intended to be concatenated to a copy of the original 
		API response.	
	
		as_dataframe=True returns a new dataframe and does not include any of 
		the unsorted entries'''

		if not as_dataframe ^ as_series:
			raise Exception('Choose as_series or as_dataframe')

		if self.sorted == False:
			raise Exception('Data has not yet been sorted')

		if as_series:
			return self.xor_bucket_series.reindex(self.raw_data.index)
		if as_dataframe:
			filtered_raw_data = self.raw_data.loc[
				self.raw_data.index.isin(self.xor_bucket_series.index)]
			filtered_raw_data['bucket_id'] = self.xor_bucket_series.values
			return filtered_raw_data

class IterativeSorter(BucketSorter):
	'''Think of these objects as single-use only'''
	def iterative_sort(self, **kwargs):
		no_collision_bucket_series = pd.Series(dtype=int)
		running_bucket_total = 0
		cycles = kwargs['cycles']

		for cycle in range(cycles):
			cycle_params = {
				param_name : iterable_param[cycle] 
					for param_name, iterable_param in kwargs.items() 
						if not param_name=='cycles'}
			try:
				self.sort(**cycle_params)
			except ValueError:
				print('Sorting halted due to low signal')
				break

			modulus = running_bucket_total + 1
			no_collision_cycle_series = self.xor_bucket_series + modulus

			number_of_new_buckets = cycle_params['num_clusters']
			running_bucket_total += number_of_new_buckets
			
			no_collision_bucket_series = pd.concat([
				no_collision_bucket_series, no_collision_cycle_series]) 

			self.inverse_prune()

		self.xor_bucket_series = no_collision_bucket_series

	def inverse_prune(self):
		self.matrix = self.matrix[self.inverted_mask]
		self.raw_data = self.raw_data[self.inverted_mask]

		self.decomposed = 	False 
		self.pruned 	=   False
		self.sorted 	= 	False

sentinel = TestingSentinel()
api_response = sentinel.req_by_days(-300, -1)
sorter = BucketSorter(api_response)
sort_args = {
	'num_clusters'			:10, 
	'svd_n'					:200, 
	'svd_target_var'		:.50,
	'euclidean_threshold'	:.9,
	'override_svd'			:True,
	'normalize'				:True}

itersort_args = {
	'cycles'				:4,
	'num_clusters'			:[5, 5, 5, 5],
	'svd_n'					:200, 
	'svd_target_var'		:.50,
	'euclidean_threshold'	:[.9, .9, .9, .9],
	'override_svd'			:True,
	'normalize'				:True}

sorter.iterative_sort(**itersort_args)
primary_result = sorter.extract(as_dataframe=True)

get = lambda k : primary_result.loc[
	primary_result['bucket_id']==k, ['bucket_id', 'title']]
pdb.set_trace()
