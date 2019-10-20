import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["soccer", "football", "pogba", "mbappe", "zlatan", "rooney", "neymar", "messi", "ronaldo", "epl", "madrid", 
	"barcelona", "psg", "manchester", "chelsea", "arsenal", "liverpool", "tottenham", "westham", "atletico", "bvb", "bayern", 
	"cristiano", "lionel", "suarez", "pele", "zidane", "hazard", "ibrahimovic", "ronaldinho", "griezmann", "salah", "iniesta", "modric",
	"bruyne", "aguero", "ramos", "rm", "lewan", "lewandowski", "beckham", "usmnt", "ozil", "pique", "countinho", "kroos", "neuer", "kante",
	"dybala", "mancity", "juv", "juventus", "barce", "milan", "galaxy", "leicester", "everton", "copa", "uefa", "ucl", "striker", "kickoff",
	"shootout", "match", "freekick"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts