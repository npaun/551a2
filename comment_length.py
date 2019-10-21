import numpy as np
import nltk

def find_comment_len(X):

	comment_lengths = np.array([])

	for x in X:
		comment_length = np.array([len(x)])
		comment_lengths = np.append(comment_lengths, comment_length)
		
	return comment_lengths