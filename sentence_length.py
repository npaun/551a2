import numpy as np
import nltk

def find_avg_sent_len(X):

<<<<<<< HEAD
	avg_sentence_lengths = np.array([])

	for x in X:
		comment_length = len(x)
		sentences = nltk.sent_tokenize(x)
		avg_len = np.array([comment_length/(len(sentences))])
		avg_sentence_lengths = np.append(avg_sentence_lengths, avg_len)
		
	return avg_sentence_lengths
=======
        res =  np.array([len(comment)/len(nltk.sent_tokenize(comment)) for comment in X])
        print(res)
        return res
>>>>>>> 36cf0cbbe1591afa1465955970fc783c59ec7fc0
