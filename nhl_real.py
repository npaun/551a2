import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["hockey", "skate", "icing", "slapshot", "ovechkin", "gretzky", "crosby", "toews", "malkin", 
	"mcdavid", "doughty", "kucherov", "karlsson", "hedman", "kopitar", "ice", "devils", "capitals", "rangers",
	"oilers", "habs", "canadiens", "leafs", "canucks", "nhlstreams", "nhl", "puck", "edmonton", "flames", "bruins",
	"caps", "sabres", "islanders", "flyers", "blackhawks", "avalanche", "coyotes", "predators", "panthers",
	"chel", "chirp", "gino", "goon", "hoser", "lettuce", "odr"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))
	
	return match_counts