import pandas
import numpy as np


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse
import sentence_length

df = pandas.read_csv('data/reddit_train.csv', header=0)
X = df['comments'].to_numpy()
Y = df['subreddits'].to_numpy()


    
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

#worse than tfidf
count_vetor = CountVectorizer(
    input='content',
    lowercase=True,
    analyzer='word',
    stop_words='english',
    ngram_range=(1,1),
    max_df=1.0,
    min_df=1,
    max_features=None,
)

avg_sentence_lengths = sentence_length.find_avg_sent_len(X)
csr_avg_sent_lens = scipy.sparse.csr_matrix(avg_sentence_lengths).transpose()

X = tfidf.fit_transform(X)
print(X.shape)

print(csr_avg_sent_lens.shape)
#X = scipy.sparse.hstack((X, csr_avg_sent_lens), format='csr')

k_fold = KFold(n_splits=5)
models = {
    'lr': LogisticRegression(),
    'mnb': MultinomialNB(),
    'decision_tree': DecisionTreeClassifier(),
    #'svm': SVC(),
    'random_forest': RandomForestClassifier(n_estimators=10)
}

for train, test in k_fold.split(X,Y):
    for model_name, model in models.items():
        model.fit(X[train], Y[train])
        print("%s score = " % (model_name),   model.score(X[test], Y[test]))