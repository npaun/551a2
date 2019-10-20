import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["naruto", "hunterxhunter", "onepunch", "akira", "mononoke", "kiki", "nausicaa", "sasuke", "kakashi",
	"evangelion", "ponyo", "fullmetal", "bebop", "steamboy", "tokyo", "japan", "nani", "animatrix", "inuyasha",
	"shippuden", "goku", "ichigo", "kurosaki", "luffy", "yagami", "uzumaki", "elric", "vegeta", "uchiha", "itachi",
	"hatake", "yeager", "lamperouge", "zoro", "ackerman", "kirito", "dragneel", "zoldyck", "mustang", "spiegel", "erza",
	"gaara", "saitama", "alucard", "himura", "anime", "asuna", "kmaina", "pikachu", "minato", "animation", "shikamaru", "jiraiya",
	"orochimaru", "tsunade", "hinata", "sakura", "kabuto"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts