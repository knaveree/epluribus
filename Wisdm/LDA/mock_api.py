import pandas as pd
import datetime as dt
import os 

class TestingSentinel:
	#Simulates some approximation of API response from database in 
	#the form of a pandas dataframe. The response represents however many
	#days worth of headlines are requested, based on a date range via
	#either negative integers counting back from -1 (following reverse
	#indexing convention) or two python datetime objects.

	def __init__(self):
		source = os.path.join(os.getcwd(), 'abcnews-date-text.csv')
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

