import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["football", "firstdown", "nfl", "packers", "seahawks", "patriots", "pats", "brady", "rodgers", "kamara", "cowboys", 
	"raiders", "rams", "steelers", "rothlisberger", "mahomes", "goff", "redskins", "49ers", "cardinals", "dolphins", "dak", 
	"prescott", "hilton", "zeke", "ezekiel", "brees", "barstool", "ab", "odell", "obj", "gronk", "manning", "gronkowski",
	"amari", "gurley", "kelce", "veon", "baker", "mayfield", "deshaun", "deandre", "tyreek", "ndamukong", "suh", "clowney", "watt", "saquon",
	"beckham", "davante", "keenan", "akiem", "ertz", "juju", "jss", "schuster", "ingram", "garrett", "jadeveon", "diggs",
	"mccaffrey", "mccoy", "jarvis", "landry", "belichick", "mcvay", "edelman", "wentz", "lockett", "chiefs", "broncos", "browns", 
	"vikings", "saints", "bears", "giants", "ravens", "chargers", "buccaneers", "colts", "jaguars", "titans", "bengals", "fumble",
	"interception", "touchdown", "td", "nfc", "afc", "harbaugh", "parcells", "gruden", "tomlin", "arians", "payton", "pederson"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts