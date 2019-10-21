import numpy as np
import nltk

def getMatches(X):

	match_counts = np.array([])
	keys = ["music", "hiphop", "country", "classical", "rock", "blues", "genre", "track", "record",
	"single", "album", "beatles", "reggae", "dreampop", "pop", "punk", "experimental", "distortion", "reverb",
	"pitch", "rhythm", "melody", "sing", "voice", "mic", "microphone", "tour", "band", "song", "producer",
	"singer", "rap", "rapper", "beat", "instrumental", "indie", "alternative", "alt", "migos", "eminem", "elvis",
	"guitar", "drum", "drums", "guitarist", "drummer", "bass", "bassist", "riff", "chords", "chord", "riffs", 
	"cd", "lyrics", "ballad", "zeppelin", "sound", "metal", "thrash", "tour", "touring", "techno", "electric", "edm", 
	"synth", "lyricist", "dance", "funk", "jazz", "remix", "billboard", "electro", "itunes", "spotify", "soundcloud",
	"radio", "headphones", "concert", "concerts", "kanye", "soundtrack"]

	for comment in X:
		words = nltk.word_tokenize(comment)
		matches = 0
		for k in keys:
			for w in words:
				if k == w.lower():
					matches += 1
		match_counts = np.append(match_counts, np.array([matches]))

	return match_counts