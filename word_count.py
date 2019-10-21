import pandas 
import numpy as np
import nltk
import xlrd
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import PorterStemmer
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def find_word_count (X):

	df = pandas.read_csv('data/reddit_train.csv', header=0)
	X = df['comments'].to_numpy()
	reddit = df['comments']
	tokenizer = RegexpTokenizer('\s+', gaps=True)
	stemmer = PorterStemmer()
	stop_words = set(stopwords.words('english'))

	strdata = reddit.values.tolist()
	tokens = [tokenizer.tokenize(str(i)) for i in strdata]
	clean_list = [] 
	for m in tokens: 
		stopped = [i for i in m if str(i).lower() not in stop_words] #remove stop words
		stemmed = [stemmer.stem(i) for i in stopped]
		clean_list.append(stemmed) #the words that are left are added to the cleaned list 
		#print(clean_list)


	count = []
	for x in clean_list: 
		count.append(len(x))
		

	
	return count 

