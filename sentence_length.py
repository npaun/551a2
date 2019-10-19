import numpy as np
import nltk

def find_avg_sent_len(X):

        res =  np.array([len(comment)/len(nltk.sent_tokenize(comment)) for comment in X])
        print(res)
        return res
