import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["mariners", "mlb", "baseball", "nationals", "reds", "marlins", "yankees", "jeter", "arod", 
	"padres", "angels", "anaheim", "giants", "pirates", "cardinals", "dodgers", "twins", "royals", "jays", 
	"phillies", "diamondbacks", "dbacks", "phils", "cubs", "sox", "redsox", "mets", "athletics", "oakland", "braves",
	 "rays", "indians", "orioles", "brewers", "marlins", "tigers", "rockies", "rangers", "expos", "pitcher", 
	 "catcher", "shortstop", "dinger", "homerun", "rbi", "single", "double", "triple", "era", "whip", "dodgers", "trout",
	  "scherzer", "verlander", "kershaw", "harper", "machado", "mookie", "giancarlo", "arenado", "lindor", "greinke", "altuve", "bregman", "yelich",
	  "ramirez", "degrom", "martinez", "machado", "votto", "stanton", "rendon", "kluber", "snell", "baez", "correa", "carpenter",
	  "rizzo", "acuna", "soto", "springer", "blackmon", "haniger", "realmuto", "bauer", "bellinger", "seager", "cano",
	  "upton", "francona", "hurdle", "maddon", "hinch", "bochy", "roberts", "cy", "bonds", "griffey", "cabrera", "mantle",
	   "dimaggio", "pujols", "ripken", "canseco", "clemente", "clemens", "koufax", "yogi", "berra", "henderson", "ortiz", "maddux", "ichiro",
	   "gehrig", "mariano", "rivera", "mattingly", "posada", "pettitte", "boone", "sabathia", "masahiro", "randy", "strikeout", "fastball", "curveball",
	   "mph", "grandslam", "meatball", "walks", "bb", "hbp", "hr", "obp", "slg", "inning"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts