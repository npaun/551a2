import pandas
import numpy as np


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
    ('cv', CountVectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# parameters = {
#     'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
# }
# array = np.array([])
# np.append(array, np.array([0]), axis=0)
# print(array)

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


#probs = sentence_length.find_avg_sent_len(comments)
# print(avg_sentence_lengths)

X = tfidf.fit_transform(comments)
print(X.shape)
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
print(X.shape)
k_fold = KFold(n_splits=5)
models = {
    'lr': LogisticRegression(),
    'mnb': MultinomialNB(),
    'bnb': BernoulliNB(),
    #'decision_tree': DecisionTreeClassifier(),
    #'svm': SVC(),
    'random_forest': RandomForestClassifier(n_estimators=10)
}

# for train, test in k_fold.split(X,Y):
#     for model_name, model in models.items():
#         model.fit(X[train], Y[train])
#         


for train, test in k_fold.split(X,Y):
    for model_name, model in models.items():

        # print("Performing grid search...")
        # print("pipeline:", [name for name, _ in pipeline.steps])
        # print("parameters:")
        # pprint(parameters)
        t0 = time()

        
        # grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=1)
        model.fit(X[train], Y[train])
        print("%s score = " % (model_name), model.score(X[test], Y[test]))


        print("done in %0.3fs" % (time() - t0))
        print()

        # print("Best score: %0.3f" % grid_search.best_score_)
        # print("Best parameters set:")
        # best_parameters = grid_search.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
