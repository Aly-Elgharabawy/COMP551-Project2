import pandas as pd
import numpy as numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn import preprocessing
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords
import csv


class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



stop_words = set(stopwords.words('english'))
data_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_train.csv',encoding='utf-8').to_numpy()
porter = PorterStemmer()
labels = data_matrix[:,2]
comments = data_matrix[:,1]
for comment in comments:
    comment = comment.lower()
    comment = porter.stem(comment)
    comment = ' '.join([word for word in comment.split() if word not in stop_words])
    for punc in string.punctuation:
        comment = comment.replace(punc,' ')
    for symbol in ['*','&','~','"']:
        comment = comment.replace(symbol,' ')
    for n in range(0,10):
        comment = comment.replace(str(n),'')

tf_idf_vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(),sublinear_tf = True)
tv = tf_idf_vectorizer.fit_transform(comments)


def get_training_data():
    #obtain data as np array and split into labels and raw comments


    #vectorize data
    training_vector = normalize(tv)

    #Encode Labels
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    training_labels = le.transform(labels)
    return training_vector, training_labels


def get_training_data_split():
    #obtain data as np array and split into labels and raw comments

    #split data
    X_train, X_test, y_train, y_test = train_test_split(comments, labels, train_size=0.8, test_size=0.2)

    #vectorize data
    training_vector = normalize(tf_idf_vectorizer.transform(X_train))
    test_vector = normalize(tf_idf_vectorizer.transform(X_test))

    #Encode Labels
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    training_labels = le.transform(y_train)
    test_labels = le.transform(y_test)

    return training_vector,test_vector, training_labels, test_labels

def get_test_data():
    #get datasets
    test_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_test.csv').to_numpy()

    
    test_comments = test_matrix[:,1]
    for comment in test_comments:
        comment = comment.lower()
        comment = porter.stem(comment)
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        comment = ' '.join([word for word in comment.split() if word not in string.punctuation])
    #vectorize based on training fit

    test_vector = normalize(tf_idf_vectorizer.transform(test_comments))

    return test_vector

def decode_labels(encoded_labels):


    #encode then decode
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    decoded_labels = le.inverse_transform(encoded_labels)

    return decoded_labels


def submit(y_pred_test):
    with open('C:/Git/COMP551-Project2/data/reddit_test.csv','r',encoding="utf8") as csvinput:
        with open('D:/reddit_pred.csv', 'w',encoding='utf8') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            i=0
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