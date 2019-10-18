import pandas as pd
import numpy as numpy
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn import preprocessing
from nltk.stem import PorterStemmer
import string
import nltk
from nltk.corpus import stopwords
import csv
from keras.preprocessing import Tokenizer

class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, comment):
         return [self.wnl.lemmatize(word) for word in word_tokenize(comment)]



stop_words = set(stopwords.words('english'))
data_matrix = pd.read_csv('C:/Git/COMP551-Project2/data/reddit_train.csv').to_numpy()
porter = PorterStemmer()
labels = data_matrix[:,2]
comments = data_matrix[:,1]
for comment in comments:
    comment = comment.lower()
    comment = porter.stem(comment)
    comment = ''.join(word for word in comment if word not in string.punctuation)
    comment = ''.join(word for word in comment if word not in stop_words)

tokenizer = Tokenizer(num_words = 44000,lower = True)
tokenizer.fit_on_texts(comments)

training_tensor = tokenizer.texts_to_sequences(comments,maxlen=200)
le = preprocessing.LabelEncoder()
le.fit(labels)
training_labels = le.transform(labels)

model = Sequential()
model.add(Embedding(44000, 100, input_length=training_tensor.shape[1]))
model.add(SpatialDropout1D(0.25))
model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25)))
model.add(Dense(20, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(training_tensor, training_labels, epochs=5, batch_size=1024,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])
