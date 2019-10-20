import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["weed", "marijuana", "bong", "joint", "bowl", "bud", "baked", "smoke", "high", "ganjga", "pot", 
	"legalize", "cheef", "cannabis", "thc", "cbd", "vape", "kush", "doja", "gram", "ounce", "eighth", 
	"doobie", "herb", "morty", "ent", "tolerance", "420", "dank", "edibles", "dab", "rig", "blaze", "blazing",
	"toke", "toking" ]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts