import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["csgo", "globaloffensive", "counterstrike", "counter", "offensive", "smokes", "nade", "nades", "ump", "dink", "dinked", "comms",
	"smoke", "flashbang", "armour", "glock", "cs", "knifing", "knifed", "knife", "competitive"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts