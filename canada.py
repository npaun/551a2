import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["canada", "trudeau", "bc", "ontario", "quebec", "alberta", "manitoba", "scotia", "nova", "brunswick", "newfoundland", "province"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts