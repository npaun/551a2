import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["daenerys", "targaryen", "arya", "stark", "gregor", "clegane", "cersei", "lannister", "tyrion", "sansa", "khal", 
	"drogo", "eddard", "joffrey", "baratheon", "petyr", "baelish", "melissandre", "sandor", "bolton", "jaime", "bran", 
	"theon", "greyjoy", "brienne", "tarth", "margaery", "tyrell", "lord", "varys", "shae", "bronn", "hodor", "jorah",
	"mormont", "ygritte", "oberyn", "missandei", "tormund", "robb", "winterfell", "westeros", "braavos", "dorne", "pentos", "meereen", "yunkai", "pentos",
	"maesters", "qarth", "kingsguard", "dosh", "khaleen", "harpy", "unsullied", "direwolf", "direwolves", "volantis", "tyrosh", "lorath",
	"pentos", "norvos", "myr", "qohor", "dragonstone", "dreadfort", "stokeworth", "cailin", "ironrath", "highpoint", "deepwood", "riverlands", 
	"riverrun", "harrenhal", "oldtown", "eyrie", "arryn", "casterly", "runestone", "castamere", "gulltown", "dornish", "sunspear", "stormlands",
	"dornishmen", "benioff", "weiss", "belfast", "hbo", "dinklage", "fairley", "maisie", "headey", "gwendoline", "hempstead", "dothraki", "faceless", "greyscale",
	"khaleesi", "maester", "rhaegar", "hllor", "sellsword", "septon", "sigil", "turncloack", "valyrian", "valar", "morghulis", "warg", "wildlings"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts