import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import nltk
from nltk.corpus import stopwords
import new_features
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


#import data as numpy
data_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_train.csv').to_numpy()
test_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_test.csv').to_numpy()



#separate labels from comments
labels = data_matrix[:,2]
comments = data_matrix[:,1]

test_comments = test_matrix[:,1]
#Perform train test split
X_train, X_test, y_train, y_test = train_test_split(comments, labels, train_size=0.8, test_size=0.2)

# X_train_new_features = np.asarray(new_features.create_features(X_train))
# X_test_new_features = np.asarray(new_features.create_features(X_test))

# test_new_features = np.asarray(new_features.create_features(test_comments))
#Vectorize data
#tf_idf_vectorizer = CountVectorizer(tokenizer = LemmaTokenizer(),binary=True)
# vectors_train = vectorizer.fit_transform(X_train)
# vectors_test = vectorizer.transform(X_test)
# vectors_all = vectorizer.transform(comments)

#Obtain TF-IDF for train and test sets
tf_idf_vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(),sublinear_tf='true')
all_idf = tf_idf_vectorizer.fit_transform(comments)
vectors_train_idf = tf_idf_vectorizer.transform(X_train)
vectors_test_idf = tf_idf_vectorizer.transform(X_test)

test_idf = tf_idf_vectorizer.transform(test_comments)

# for new_feature in X_train_new_features:
#     vectors_train_idf = sparse.hstack((vectors_train_idf,new_feature[:,None]))

# for new_feature in X_test_new_features:
#     vectors_test_idf = sparse.hstack((vectors_test_idf,new_feature[:,None]))

# for new_feature in test_new_features:
#     vectors_test_idf = sparse.hstack((test_idf,new_feature[:,None]))

vectors_train_idf = normalize(vectors_train_idf)
vectors_test_idf = normalize(vectors_test_idf)
test_idf = normalize(test_idf)

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
y_all = le.transform(labels)

clf = LogisticRegression(penalty = 'l1',C=5.6,max_iter=700)
lr = clf.fit(all_idf, y_all)
model = SelectFromModel(lr,prefit=True)
y_pred = clf.predict(vectors_test_idf)
print(metrics.accuracy_score(y_test, y_pred))
y_pred_test = clf.predict(test_idf)

vectors_train_idf = model.transform(vectors_train_idf)
vectors_test_idf = model.transform(vectors_test_idf)
clf = MultinomialNB(alpha=0.05)
clf.fit(all_idf,y_all)
y_pred_test = clf.predict(test_idf)
y_pred_test = le.inverse_transform(y_pred_test)
i = 0
with open('C:/Git/COMP551-Project2/data/reddit_test.csv','r',encoding="utf8") as csvinput:
    with open('D:/reddit_pred.csv', 'w',encoding='utf8') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        towrite = []
        row = next(reader)
        row.append('subreddits')
        towrite.append(row)

        for row in reader:
            row.append(y_pred_test[i])
            towrite.append(row)
            i+=1
        writer.writerows(towrite)
print('done')

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(vectors_train_idf,y_train)
y_pred = (clf.predict(vectors_test_idf))
print(metrics.accuracy_score(y_test, y_pred))


