#'''
#News API as default testing framework
#
#*Keep author
#*Keep content (duh)
#*Keep description?
#*Keep title
#*Url
#*Timestamp
#*Source
#
#Write the database parser to accept a variable callback function
#specific to each API response (preprocessing stage)
#
#Basic Article Schema
#
#import id
#import source
#
#def parse_source(title):
#	return source
#schema_dict = {
#	'id': hash(name, timestamp)
#	'author' : #author if author else parse_source(title)
#	'name': source if source=='Google' else parse_source(title), 
#	'timestamp': #either grabbed from schema or from our call time,
#	'content' : content,
#}
#'''
API_KEY = '4ac64df715f2498bb6d7590326907083'

#So this is a copy-pastable variant of what Jonathan sent earlier, formatted
#as a flattened python dictionary. The other half, the daily master, I'm real-
#izing will be constructed dynamically from a collection of these article
#dictionaries (schemas). So what Jonathan's API will spit out will probably 
#need to look something like a collection of these, in particular all the
#articles captured for a given batch. I'll have to build the Master from
#these articles. 

#I'm not sure exactly how you'll want to package the response from the API,
#Jonathan, but I'm voting for a list of these dictionaries. 

#I've introduced a few new things, namely the Gregorian Ordinal date, as
#what I think is the most straightforward way to group a day's batch-- since
#by definition they'll be occuring within a given day. (This ordinal is just
#the integer number of days since the Romans nailed that nice young man
#to a cross. Super weird that's what all of the computers in the world 
#count from...) Any native python datetime object does this with .toordinal(). 

#I'm recommending we hash a tuple of (source, #other) for some other detail
#to product a record's unique article ID. This allows for tracing back 
#if we have those two datums, and easily guarantees uniqueness. The other 
#detail could be author or source, doesn't really matter. 

#The timestamp entry may be superfluous, given that the table_id contains 
#a 1 to 1 date reference (the Gregorian Ordinal date). Time of day for an 
#article is probably not relevant in the overwhelming majority of cases, and
#analysis specifically of by-the-hour breaking news (where it would matter)
#is beyond the scope of what we can even accomplish with a 24 hour batch 
#approach.

#Finally, regarding the description entry: I'm not sure if this means our own 
#description or a source-provided description. I'm certainly unsure if it's 
#necessary or efficient to introduce our own summary of an article at this 
#stage, given that many of the articles won't make it out of the topic sorting 
#algorthm. On the other hand, if the descriptions are provided by the sources, 
#I agree that this is good data we should preserve for later, but we may run
#into trouble if descriptions aren't universally available. 

example_article_dictionary = {
	'id' : 		   None,	#Gregorian Ordinal date (wrt Greenwich UK) (int)
	'article_id' : None, 	#Hash of title plus some other detail (int)
	'source' : 	   None, 	#Source name (string)
	'content': 	   None, 	#Unvectorized string of content (string)
	'description': None,	#Our own AI generated pre-summary? (string)
	'title' :	   None, 	#Unvectorized string of title (string)
	'timestamp':   None,	#Timestamp (Python's native datetime.datetime obj)
}

#Anyway, let me know if I'm missing something, or if this doesn't work for some
#reason I'm not seeing. Otherwise I'm going to move ahead assuming the data
#I get from the API will be formatted as some kind of collection of these 
#dictionaries... though I would prefer that collection to be a python list.


example_daily_master = {
	'id' : Julian Date
}

	


