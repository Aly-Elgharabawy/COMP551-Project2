import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


#import data as numpy
data_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_train.csv').to_numpy()

#separate labels from comments
labels = data_matrix[:,2]
comments = data_matrix[:,1]

#Perform train test split
X_train, X_test, y_train, y_test = train_test_split(comments, labels, train_size=0.8, test_size=0.2)


#Vectorize data
vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

#Obtain TF-IDF for train and test sets
tf_idf_vectorizer = TfidfVectorizer()
vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
vectors_test_idf = tf_idf_vectorizer.transform(X_test)

vectors_train_normalized = normalize(vectors_train)
vectors_test_normalized = normalize(vectors_test)
le = preprocessing.LabelEncoder()
le.fit(y_train)
le.transform(y_train)

clf = LogisticRegression()

clf.fit(vectors_train_normalized, y_train)
y_pred = clf.predict(vectors_test_normalized)
print(metrics.accuracy_score(y_test, y_pred))

clf = MultinomialNB(alpha=.01)
scores = cross_val_score(clf, vectors_train_normalized, y_train, cv=5)
print(scores)