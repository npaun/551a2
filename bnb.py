import pandas
import numpy as np
import nltk
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import operator




class BernoulliNaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, X, Y, to_features=lambda X: X):
        # Sort to get all samples in order by class
        Y_sort = np.lexsort((Y,))
        X, Y = X[Y_sort], Y[Y_sort]
        X = to_features(X)

        # Priors
        self.class_name, class0, classn = np.unique(Y, return_index=True, return_counts=True)
        self.priors = classn / Y.shape[0] # P(subreddit) = # in subreddit / # total

        # Group samples by class
        X_by_class = [X[c0:c0+n] for c0, n in zip(class0, classn)]
        
        # Per word, number of comments with this word
        count_by_class = [np.sum(Xc, axis=0) for Xc in X_by_class]
        # Dividing by number of comments in subreddit, obtain P(word | subreddit)
        self.likelihood = [count_c / n for count_c, n in zip(count_by_class, classn)]

        # Evidence
        self.evidence = np.sum(count_by_class, axis=0) / Y.shape[0]
        print("Fitting completed.")

    def predict(self, Xtest, to_features=lambda X: X):
        # Here we don't need to control ordering but handle transform for symmetry
        Xtest = to_features(Xtest)
        return [max(self._prob(x, cls)
                    for cls in range(len(self.priors))) 
                for x in Xtest]

    def score(self, Ypredict, Ytest):
        correct, _ = np.where(Ypredict == Ytest)
        return len(correct) / len(Ytest)

    def _prob(self, x, cls):
        print("XXX", x)
        likelihood = self._bernoulli_from_features(x, self.likelihood[cls])
        evidence = self._bernoulli_from_features(x, self.evidence)
        return self.priors[cls]*evidence/likelihood

    def _bernoulli_from_features(self, x, p):
        return np.where(x == 1, p, 1 - p)



def word_present(count_vec):
    return lambda X: count_vec.fit_transform(X).toarray()

def main():
    df = pandas.read_csv('data/reddit_train.csv', header=0)
    X = df['comments'].to_numpy()
    Y = df['subreddits'].to_numpy()

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

    k_fold = KFold(n_splits=5)
    for train, test in k_fold.split(X, Y): 
        model = BernoulliNaiveBayes()
        model.fit(X[train], Y[train], to_features=word_present(count_vec))
        Ypredict = model.predict(X[test], to_features=word_present(count_vec))
        accuracy = model.score(Ypredict, Y[train])
        print("npaun BNB = ", accuracy)

if __name__ == '__main__':
    main()

