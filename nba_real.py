import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["basketball", "hoop", "rim", "jumpshot", "dribble", "nba", "court", "lakers", "cavs",  "celtics", "knicks", 
		"raptors", "pistons", "pacers", "dunk", "sixers", "nets", "kyrie", "durant", "lebron", "bryant", "curry", 
		"harden", "cp3", "westbrook", "kawhi", "ad", "giannis", "damian", "pg", "embiid", "klay", "draymond", "lowry", 
		"fg%", "3p%", "ppg", "shaq", "birdman", "oladipo", "aldridge", "hayward", "woj", "wojnarowski", "siakam", "melo",
		"wade", "dwade", "kanter", "bron", "mj", "kobe", "iverson", "kd", "wilt", "kareem", "duncan", "nash", "dirk", 
		"tmac", "kanter", "spurs", "nuggets", "mavs", "clippers", "thunder", "pelicans", "pels", "blazers", "hornets",
		"carmelo"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))
	
	return match_counts