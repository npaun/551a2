import pandas
import numpy as np
import nltk
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import operator




class BernoulliNaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, X, Y, all_classes):
        Y_sort = np.lexsort((Y,))
        X, Y = X[Y_sort], Y[Y_sort]

        # Priors
        self.class_name, class0, classn = np.unique(Y, return_index=True, return_counts=True)
        
        missing_classes = set(all_classes) - set(self.class_name)
        self.class_name = np.append(self.class_name, np.array([mc for mc in missing_classes]))
        class0 = np.append(class0, np.array([0 for _ in missing_classes], dtype=int))
        classn = np.append(classn, np.array([0 for _ in missing_classes], dtype=int))

        self.priors = np.log(classn / Y.shape[0]) # P(subreddit) = # in subreddit / # total
        # Group samples by class
        X_by_class = [X[c0:c0+n] for c0, n in zip(class0, classn)]
        
        # Per word, number of comments with this word
        count_by_class = [np.sum(Xc, axis=0) for Xc in X_by_class]

        # Dividing by number of comments in subreddit, obtain P(word | subreddit)
        self.likelihood = [(count_c + 1) / (n + 2) for count_c, n in zip(count_by_class, classn)]
        self.log_p = [np.log(l_c) for l_c in self.likelihood]
        self.log_np = [np.log(1 - l_c) for l_c in self.likelihood]

        # Evidence
        #self.evidence = np.sum(X, axis=0) / X.shape[0]
        print("\t# fitted")

    def predict(self, Xtest):
        return np.array([self._classify(x) for x in Xtest])

    def score(self, Xraw, Ypredict, Ytest):
        correct = np.count_nonzero(np.equal(Ypredict, Ytest))
        for comment, guess, real in zip(Xraw, Ypredict, Ytest):
            print("%s -> guess %s: real %s" % (comment, guess, real))

        return correct / Ytest.shape[0]

    def _classify(self, x):
        guesses = [self._class_estimate(x,cls) for cls in range(len(self.priors))]
        best =  np.argmax(np.array(guesses))
        return self.class_name[best]

    def _class_estimate(self, x, cls):
        if np.all(x == 0):
            return 0

        likelihood = np.sum(np.where(x == 1, self.log_p[cls], self.log_np[cls])) 

        return self.priors[cls] + likelihood

    def _bernoulli_from_features(self, x, p):
        return np.sum(np.where(x == 1, p, 1 - p))



def word_present(X):
    count_vec = CountVectorizer(
        input='content',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1,1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        binary=True # WARNING: Model will explode if you remove this
    )

    return count_vec.fit_transform(X).toarray()

def main():
    df = pandas.read_csv('data/reddit_train.csv', header=0)
    X = df['comments'].to_numpy()
    Y = df['subreddits'].to_numpy()
    print("# loaded")
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Sort to get all samples in order by class
    X_raw = X[:]
    X = word_present(X)
    all_classes = np.unique(Y)
    for comment, features in zip(X_raw, X):
        if np.all(features == 0):
            print("Comment is unusable -- will have random prediction: ", comment)

    
    print("# transformed")
    for train, test in k_fold.split(X, Y): 
        model = BernoulliNaiveBayes()
        model.fit(X[train], Y[train], all_classes)
        Ypredict = model.predict(X[test])
        print(Ypredict)
        print(Y[test])
        accuracy = model.score(X_raw[test], Ypredict,  Y[test])
        print("npaun BNB = ", accuracy)

if __name__ == '__main__':
    main()

