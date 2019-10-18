import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn import svm
import data

#Get full and split data
print('data1')
training_vector, training_labels = data.get_training_data()
print('data2')
training_vector_split, test_vector_split, training_labels_split, test_labels_split = data.get_training_data_split()

test_vector = data.get_test_data()



print("Step 2")

#Create Logistic Regression with full dataset to feature select using l1 regularization



clf0 = LogisticRegression(penalty = 'l1',C=4,max_iter=10000)
lr = clf0.fit(training_vector,training_labels)


model = SelectFromModel(lr,prefit=True)

training_vector = model.transform(training_vector)
training_vector_split = model.transform(training_vector_split)
test_vector_split = model.transform(test_vector_split)
test_vector = model.transform(test_vector)





# clf2 = svm.SVC(C=0.32,kernel='linear',probability = True)
# clf2.fit(training_vector,training_labels_split)
# pred = clf2.predict(test_vector)
# pred = data.decode_labels(pred)
# data.submit(pred)


clf1 = MultinomialNB(alpha = 0.05) 
clf2 = MultinomialNB(alpha = 0.07) 
clf3 = LogisticRegression(penalty = 'l2', C = 3,max_iter = 250,n_jobs = -1)
clf4 = LogisticRegression(penalty = 'l2', C = 3,max_iter = 250,n_jobs = -1)
clf5 = SGDClassifier(loss = 'log',alpha=0.0001,max_iter=500,n_jobs=-1)
clf6 = SGDClassifier(loss = 'log',penalty='l1',alpha=0.001,max_iter=500,n_jobs=-1,learning_rate = 'constant',eta0=0.001)



eclf1 = VotingClassifier(estimators=[('mnb' , clf1), ('lr', clf3),('lr2', clf4),('sgd',clf5),('sgd2',clf6),
('mnb2',clf2)], voting='soft')

eclf1.fit(training_vector_split,training_labels_split)
pred = eclf1.predict(test_vector_split)
print(metrics.accuracy_score(test_labels_split,pred))