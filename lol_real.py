import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["rammus", "ashe", "tristana", "caitlyn", "malphite", "kaisa", "rengar", "kha", "summoner", 
	"rift", "malzahar", "orianna", "ivern", "warwick", "imaqtpie", "bjergsen", "ahri", "akali", "ecko", 
	"aatrox", "azir", "annie", "bard", "blitzcrank", "brand", "amumu", "anivia", "braum", "support", "bot",
	 "jungler", "camille", "mundo", "evelynn", "ezreal", "fizz", "fiddlesticks", "gnar", "gragas", "galio", 
	 "gangplank", "garen", "hec", "hecarim", "singed", "illaoi", "fid", "graves", "jhin", "jinx", "kayn", "maoki",
	  "lulu", "lux", "lissandra", "riot", "pyke", "poppy", "riven", "sivir", "talon", "dom", "riot", "rekkles", "vayne",
	  "lb", "ziggs", "jax", "lcs", "soaz", "yellowstar", "xpeke", "youngbuck", "perkz", "faker", "skt", "worlds",
	  "shaco", "zilean", "leagueoflegends", "sonya", "shyvana", "shyv", "sion", "sylas", "soraka", "raka", "teemo",
	  "taric", "thresh", "renek", "renekton", "backdoor", "rumble", "rek", "sai", "udyr", "urgot", "varus", "voli",
	   "volibear", "vel", "xayah", "xerath", "xin", "yas", "yasou", "yorick", "zed", "zyra", "zoe", "zilean", "ali",
	   "alistar", "qiyana", "yummi", "neeko", "kai", "ornn", "kled", "taliyah", "aurelion", "kindred"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts