import pandas
import numpy as np


from pprint import pprint
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.sparse
import gensim.downloader
import nba_real
import nhl_real
import lol_real
import trees_real
import overwatch
import soccer
import nfl
import got
import baseball
import canada
import csgo
import anime
import music
import comment_length
import sentence_length
#import linker

#w2v = gensim.downloader.load('glove-twitter-25')
df = pandas.read_csv('data/reddit_train.csv', header=0)
print(df)
comments = df['comments'].to_numpy()
X = comments


Y = df['subreddits'].to_numpy()

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


tfidf = TfidfVectorizer(input='content', 
    strip_accents=None,
    lowercase=True,
    analyzer='word',
    stop_words='english',
    ngram_range=(1,1),
    max_df=1.0,
    min_df=1,
    max_features=None,
)
anal = tfidf.build_analyzer()
#res = anal(comments)


#worse than tfidf
count_vector = CountVectorizer(
    input='content',
    lowercase=True,
    analyzer='word',
    stop_words='english',
    strip_accents='ascii',
    ngram_range=(1,1),
    max_df=1.0,
    min_df=1,
    max_features=None,
)


pipeline = Pipeline([
    ('cv', count_vector),
    ('clf', MultinomialNB()),
])

class CrapInjector(object):
    def __init__(self, comments):
        self.comments = comments
    def fit(self,X,Y=None):
        return self

    def transform(self, X, Y=None):
        comments = self.comments
        avg_sentence_lengths = sentence_length.find_avg_sent_len(comments)
        csr_avg_sent_lens = scipy.sparse.csr_matrix(avg_sentence_lengths).transpose()

        comment_lengths = comment_length.find_comment_len(comments)
        csr_comment_lens = scipy.sparse.csr_matrix(comment_lengths).transpose()

        nba_matches = nba_real.getMatches(comments)
        csr_nba_matches = scipy.sparse.csr_matrix(nba_matches).transpose()

        nhl_matches = nhl_real.getMatches(comments)
        csr_nhl_matches = scipy.sparse.csr_matrix(nhl_matches).transpose()

        lol_matches = lol_real.getMatches(comments)
        csr_lol_matches = scipy.sparse.csr_matrix(lol_matches).transpose()

        trees_matches = trees_real.getMatches(comments)
        csr_trees_matches = scipy.sparse.csr_matrix(trees_matches).transpose()

        soccer_matches = soccer.getMatches(comments)
        csr_soccer_matches = scipy.sparse.csr_matrix(soccer_matches).transpose()

        overwatch_matches = overwatch.getMatches(comments)
        csr_overwatch_matches = scipy.sparse.csr_matrix(overwatch_matches).transpose()

        nfl_matches = nfl.getMatches(comments)
        csr_nfl_matches = scipy.sparse.csr_matrix(nfl_matches).transpose()

        got_matches = got.getMatches(comments)
        csr_got_matches = scipy.sparse.csr_matrix(got_matches).transpose()

        baseball_matches = baseball.getMatches(comments)
        csr_baseball_matches = scipy.sparse.csr_matrix(baseball_matches).transpose()

        canada_matches = canada.getMatches(comments)
        csr_canada_matches = scipy.sparse.csr_matrix(canada_matches).transpose()

        csgo_matches = csgo.getMatches(comments)
        csr_csgo_matches = scipy.sparse.csr_matrix(csgo_matches).transpose()

        anime_matches = anime.getMatches(comments)
        csr_anime_matches = scipy.sparse.csr_matrix(anime_matches).transpose()

        music_matches = music.getMatches(comments)
        csr_music_matches = scipy.sparse.csr_matrix(music_matches).transpose()


        #X = scipy.sparse.hstack((X, csr_avg_sent_lens), format='csr')
        #X = scipy.sparse.hstack((X, csr_comment_lens), format='csr')
        X = scipy.sparse.hstack((X, csr_nba_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_nhl_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_lol_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_trees_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_soccer_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_overwatch_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_nfl_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_got_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_baseball_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_canada_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_csgo_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_anime_matches), format='csr')
        X = scipy.sparse.hstack((X, csr_music_matches), format='csr')
        return X

avg_sentence_lengths = sentence_length.find_avg_sent_len(comments)
csr_avg_sent_lens = scipy.sparse.csr_matrix(avg_sentence_lengths).transpose()

comment_lengths = comment_length.find_comment_len(comments)
csr_comment_lens = scipy.sparse.csr_matrix(comment_lengths).transpose()
X = tfidf.fit_transform(comments)
print(X.shape)
print(csr_avg_sent_lens.shape)
X = scipy.sparse.hstack(X, csr_avg_sent_lens)
X = scipy.sparse.hstack(X, csr_comment_lens)

k_fold = KFold(n_splits=5)
models = {
    'lr': LogisticRegression(),
    'mnb': MultinomialNB(),
    #'bnb': BernoulliNB(),
    #'decision_tree': DecisionTreeClassifier(),
    #'svm': SVC(),
    'random_forest': RandomForestClassifier(n_estimators=10)
}

for train, test in k_fold.split(X, Y):
    for model_name, model in models.items():
        model.fit(X[train], Y[train])
        print("%s score = " % (model_name), model.score(X[test], Y[test]))

# for model_name, model in models.items():
#     pipeline = Pipeline([
#         ('CountVectorizer', count_vector),
#         ('CrapInjector', CrapInjector(X)),
#         ('Classifier_' + model_name, model)])

#     parameters = {
#             'CountVectorizer__ngram_range': [(1,1),(2,2),(3,3)],
#             'CountVectorizer__binary': [True, False],
#             'CountVectorizer__stop_words': ['english', None],
#     }


    
#     grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=1)
#     grid_search.fit(X, Y)
#     print("%s score = " % (model_name), grid_search.score(X, Y))


#     try:
#         print("Best parameters set:")
#         best_parameters = grid_search.best_estimator_.get_params()
#         for param_name in sorted(parameters.keys()):
#              print("\t%s: %r" % (param_name, best_parameters[param_name]))
#     except:
#         pass


#     pprint(grid_search.cv_results_)

