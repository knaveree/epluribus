import pytest
import pdb
import pandas as pd
import os
import pickle
from LDA_model import LDAModel
from mock_api import TestingSentinel


sentinel = TestingSentinel()

with open(os.path.join(os.getcwd(), 'modelpack.pkl'), 'rb') as file:
	model = pickle.load(file)

response = sentinel.req_by_days(-300)
sortable_series = response['title']
model.bucket_sort(sortable_series)

primary_result = response
primary_result['bucket_id'] = model.category_series()

get = lambda k : primary_result.loc[
	primary_result['bucket_id']==k, ['bucket_id', 'title']]

pdb.set_trace()

