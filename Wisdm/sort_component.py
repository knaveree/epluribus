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

testing_sentinel = TestingSentinel()
class BucketSorter:
	def __init__(self, start=-1, end=-1, sentinel=testing_sentinel):
		if not isinstance(start, type(end)):
			raise Exception('Invalid input date format')
		self.range_duple = (start, end)

		self._call_sentinel(sentinel=sentinel)

		self.vectorizer = TfidfVectorizer()
		func = self._preprocess_each_title
		processed_titles = self.raw_data['title'].apply(func)
		self.matrix = self.vectorizer.fit_transform(processed_titles)

		self.decomposed = 	False 
		self.pruned 	=   False
		self.sorted 	= 	False

	def _call_sentinel(self, sentinel=testing_sentinel):
		start, end = self.range_duple
		if isinstance(start, dt.datetime): 
			self.raw_data = testing_sentinel.req_by_dto(start, end)	
		elif isinstance(start, int):
			self.raw_data = testing_sentinel.req_by_days(start, end)	

	def _preprocess_each_title(self, title):
		words = word_tokenize(title.lower())
		words = [w for w in words if w.isalpha() and w not in stop_words]
		return ' '.join(words)			

	def svd(self, n=100, target_variance=.95, override=False): 
		t = target_variance
		self.svd = TruncatedSVD(n_components=n)
		matrix_reduced = self.svd.fit_transform(self.matrix)
		v = self.svd.explained_variance_ratio_.sum()
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
		self.df_svd_reduced['bucket_id'] = kmeans.labels_

	def prune_records(self, k_records=False, percentile_cutoff=False):
		if self.decomposed == False:
			raise Exception('Run BucketSorter().svd method before pruning')
		if not k_records ^ percentile_cutoff:
			raise Exception('Select k_records xor percentile_cutoff')

		self.df_svd_reduced['EuclideanNorm'] = self.df_svd_reduced.apply(
			lambda row : np.sum((row.values)**2),
			axis=1)
		sorted_df = self.df_svd_reduced.sort_values(by='EuclideanNorm')

		if not percentile_cutoff==False:
			m_records = len(sorted_df)
			k_records = ((100 - percentile_cutoff) * m_records) // 100

		self.df_svd_reduced = sorted_df[sorted_df.index < k_records]
		self.matrix_reduced = self.df_svd_reduced.values
		self.pruned = True

	def sort(
		self, 
		num_clusters=10, 
		svd_n=50, 
		k_records=False,
		percentile_cutoff=False,
		svd_target_var=.95,
		override_svd=False):

		self.svd(n=svd_n, target_variance=svd_target_var, override=override_svd)
		if k_records or percentile_cutoff:	
			self.prune_records(
				k_records=k_records, 
				percentile_cutoff=percentile_cutoff)
		self.k_means(num_clusters)
		self.sorted = True

	def extract(self, as_series=False, as_dataframe=False):
		if not as_dataframe ^ as_series:
			raise Exception('Choose as_series or as_dataframe')

		#as_series=True provides the Xor bucket values matched to each
		#article_id, and includes null values for unassignged articles.
		#This is intended to be concatenated to a copy of the original 
		#API response.	
	
		#as_dataframe=True provides a new table and does not include any of 
		#the unsorted entries

		if self.sorted == False:
			raise Exception('Data has not yet been sorted')

		xor_bucket_series = self.df_svd_reduced['bucket_id']
		master_idx = self.raw_data.index

		if as_series:
			return xor_bucket_series.reindex(self.raw_data.index)
		if as_dataframe:
			filtered_raw_data = self.raw_data.loc[
				self.raw_data.index.isin(xor_bucket_series.index)]
			filtered_raw_data['bucket_id'] = xor_bucket_series.values
			return filtered_raw_data

sorter = BucketSorter(-14, -1)
sorter.sort(
	num_clusters=20, 
	svd_n=25, 
	k_records=False,
	percentile_cutoff=10,
	svd_target_var=.95,
	override_svd=True)
pdb.set_trace()
sample = result.loc[result['bucket_id'] == 0, ['title', 'bucket_id']]
result = sorter.extract(as_dataframe=True)
print(result)
