import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["oxton", "overwatch", "ow", "widowmaker", "dva", "hanzo", "sombra", "doomfist", "mei", "genji", "winston", 
	"moira", "symmetra", "brigitte", "brig", "sym", "blizzard", "reaper", "trickle", "qp", "projectile", "koth", 
	"z9", "akm", "pvp", "pve", "bos", "bastion", "orisa", "goats", "gr", "hitscan", "pug", "capped",
	"pocketing", "boop", "zarya", "lucio", "bongo", "beyblade", "ana", "wallhack", "boostio", "2cp", "rein", "reinhardt", "widow", "zen",
	"zenyatta", "hog", "gengu", "roadhog", "junkrat", "gengi", "torbjorn", "genji", "76", "pharah", "pharmacy", "handsoap"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts