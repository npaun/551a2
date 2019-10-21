# This is our best sklearn model, isolated from all of the experiments we were trying
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas

### Load comments ####

df_train = pandas.read_csv('data/reddit_train.csv', header=0)
X_train = df_train['comments'].to_numpy()
Y_train = df_train['subreddits'].to_numpy()


df_test = pandas.read_csv('data/reddit_test.csv', header=0)
X_test = df_test['comments'].to_numpy()
ID_test = df_test['id'].to_numpy()

tfidf = TfidfVectorizer(input='content',
        strip_accents=None,
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1,1),
        max_features=None
)

mnb = MultinomialNB()

pipeline = Pipeline([
        ('bow', tfidf),
        ('clf', mnb)])

pipeline.fit(X_train, Y_train)
score = pipeline.score(X_train, Y_train)
print("Final score (train/train)", score)

Y_predicted = pipeline.predict(X_test)
df_predicted = pandas.DataFrame({'id': ID_test, 'Category': Y_predicted})
df_predicted.to_csv('data/results.csv', header=True, index=False)
