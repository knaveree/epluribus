import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class UserAnalyzer:
	'''Analyzer class for user posts. Requires no parameters to initialize.

	This component is intended to work as a passthrough object for every new
	post before the post is archived to the database. Each post gets scored
	for sentiment and idealogy (idealogy feature currently not implemented) 
	and after every post a user's score is updated and, if the score crosses 
	defined thresholds, their sentiment and idealogy labels will change. 

	Public methods on this class include: 
		.update() to add new posts and possibly 
		.get_state(user_id=False, post_id=False, sentiment=True, ideology=True) 
			to retrieve the sentiment/ideology score of a given post or user by 
			post_id or user_id '''
	
	sia = SentimentIntensityAnalyzer()
	roast_labels = {
		(1,  .4)  : 'Teacher\'s Pet',
		(.4, .2)  : 'Goodie Two Shoes',
		(.2,-.2)  : 'Basic Bitch',
		(-.2,-.4) : 'Wet Blanket',
		(-.4, 1)  : 'Doom Troll'}

	bias_labels = {
		(1,  .4)  : 'Revolution? Cool story, bro',
		(.4, .2)  : 'Biden\'s Bitch',
		(.2,-.2)  : 'Bruh, Pick A Side',
		(-.2,-.4) : 'Trump Troll',
		(-.4, 1)  : 'That an AR-15? Or you just happy to see me?'}
	
	def unique_id(self, *args, digits=8):
		if not isinstance(digits, int):
			raise Exception('digits parameter must be an integer')

		if digits > 18:
			digits = 18
		elif digits < 0:
			digits = 0

		hash_input = tuple(args)
		as_string = str(hash(hash_input))[0:8]
		return int(as_string)

	def __init__(self):
		return None

	def _pull(self, user_id):
		pass

	def _push(self, post_update=False, user_update=False):
		if post_update != False:
			#Should contain code to interact with PostgreSQL database
			pass
		if user_update != False:
			#Should contain code to interact with PostgreSQL database
			pass
	
	def _populate_post_record(self, **kwargs):
		user_id   = kwargs['user_id']
		tstamp    = kwargs['tstamp']
		post_body = kwargs['post_body']

		post_record = {
			'post_id' 		: self.unique_id(user_id, tstamp),
			'user_id' 		: user_id,
			'post_body' 	: post_body, 
			'tstamp'	 	: tstamp,
			'post_sentiment': self._score_sentiment(post_body)}
			#'post_bias:	: self._score_idealogies(post_body)}

		return post_record

	def _populate_user_update(self, **kwargs):
		state = self._pull(user_id)
		n 	  = state['post_count'] + 1

		sentiment_score = state['user_sentiment_score']
		ideology_score  = state['user_ideology_score']

		new_sentiment_score = sentiment_score/n + post_record['post_sentiment']
		new_ideology_score  = ideology_score/n  + post_record['post_ideology']
		roast_label		 	= self._roast_labeler(new_sentiment_score)
		bias_label 			= self._bias_labeler(new_ideology_score)
			
		user_update = {
			'user_id' 	 		   	: user_id,
			'post_count' 		   	: n,
			'user_sentiment_score' 	: new_sentiment_score,
			'user_ideology_score'  	: new_ideology_score,
			'roast_label' 		 	: roast_label,
			'bias_label' 			: bias_label}
		}

		return user_update

	def update(self, **kwargs):
		user_id = kwargs['user_id']
		body 	= kwargs['post_body']
		tstamp 	= dt.datetime.now()

		post_record = self._populate_post_record(
			body		= body,
			user_id		= user_id,
			tstamp 	= tstamp)

		user_update = self._populate_user_update(
			user_id		= user_id, 
			post_record = post_record)

		self._push(
			user_update = user_update,
			post_update = post_record)

		return None

	def _score_sentiment(self, post_body):	
		return self.sia(post_body)['compound']

	def _score_idealogies(self, post_body):
		'''Feature under development'''
		pass
	
	def _roast_labeler(self, user_score, post_count):
		for roast_label, interval in self.roast_labels.items():
			left_bound, right_bound = interval
			if left_bound <= user_score < right_bound:
				return roast_label
		raise Exception('Score error: outside normal boundaries')
	
	def _bias_labeler(self, user_score, post_count):
		for bias_label, interval in self.bias_labels.items():
			if interval[1] <= user_score < interval[0]:
				return bias_label 
		raise Exception('Score error: outside normal boundaries')

#postgresql is database standard moving forward

