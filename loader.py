import pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse

import linker
df = pandas.read_csv('data/reddit_train.csv', header=0)
comments = df['comments'].to_numpy()
Y = df['subreddits'].to_numpy()

subreddits = []
domains = []
for i, txt in enumerate(comments):
    comments[i], sr, ds = linker.find_links(txt)
    subreddits.append(sr)
    domains.append(ds)

import pprint
pprint.pprint(domains)
print(linker.known_subreddits)
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

comments_tfidf = tfidf.fit_transform(comments)
ksr = sorted(list(linker.known_subreddits))
mlbin = MultiLabelBinarizer(sparse_output=True)
subreddits_1hot = mlbin.fit_transform(subreddits)
mlbin = MultiLabelBinarizer(sparse_output=True)
domains_1hot = mlbin.fit_transform(domains)
print(comments_tfidf, comments_tfidf.shape)
print(subreddits_1hot, subreddits_1hot.shape)
X = scipy.sparse.hstack((comments_tfidf, subreddits_1hot, domains_1hot),format='csr')
print(X,X.shape)

k_fold = KFold(n_splits=5)
models = {
    #'lr': LogisticRegression(),
    'mnb': MultinomialNB(),
    'bnb': BernoulliNB(binarize=True),
    #'decision_tree': DecisionTreeClassifier(),
    #'svm': LinearSVC(),
    #'random_forest': RandomForestClassifier(n_estimators=10)
}



for train, test in k_fold.split(X,Y):
    for model_name, model in models.items():
        model.fit(X[train], Y[train])
        print("%s score = " % (model_name),   model.score(X[test], Y[test]))
