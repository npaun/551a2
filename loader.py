import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

df = pandas.read_csv('data/reddit_train.csv', header=0)
X = df['comments'].to_numpy()
Y = df['subreddits'].to_numpy()

tfidf = TfidfVectorizer(input='content', 
    strip_accents=None,
    lowercase=True,
    analyzer='word',
    stop_words='english',
    ngram_range=(2,2),
    max_df=1.0,
    min_df=1,
    max_features=None,
)

X = tfidf.fit_transform(X)

k_fold = KFold(n_splits=5)
mnb = MultinomialNB()
for train, test in k_fold.split(X,Y):
    mnb.fit(X[train], Y[train])
    print("MNB score = ",   mnb.score(X[test], Y[test]))

print(X)
