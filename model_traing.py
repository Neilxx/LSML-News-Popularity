import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingRegressor
import csv

X = pd.read_csv('train_feature02.csv')
X_test = pd.read_csv('test_feature02.csv', encoding = "ISO-8859-1")
X_both = X.append(X_test)
X = 0
X_test = 0
X_both_one_hot = pd.get_dummies(X_both)
X_both = 0
#X_one_hot = X_both_one_hot[:27643][:]
#X_test_one_hot = X_both_one_hot[27643:][:]
#X_both_one_hot = 0
y = pd.read_csv('popularity.csv')
y = y['Popularity']

pipe = Pipeline([('std', StandardScaler()),
                 ('clf', LogisticRegression())])
#                 ('clf', KNeighborsClassifier(n_neighbors=10))])
#               DecisionTreeClassifier(max_depth=3, random_state=0)

pipe.fit(X_both_one_hot[:27643][:], y)
y_pred = pipe.predict_proba(X_both_one_hot[:27643][:])[:,1]
#y_pred = pipe.predict_proba(X_both_one_hot[27643:][:])[:,1]
'''

pipe1 = Pipeline([('std', StandardScaler()), ('clf', DecisionTreeClassifier(max_depth=6, random_state=0))])
pipe2 = Pipeline([('std', StandardScaler()), ('clf', GradientBoostingClassifier(n_estimators = 20))])

clf = VotingClassifier(estimators=[('dt', pipe1), ('lr', pipe2)],
                           voting='soft', weights=[1,2])
clf.fit(X_both_one_hot[:27643][:], y)
y_pred = clf.predict(X_both_one_hot[27643:][:])
'''
with open('f_LR_input02.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Popularity'])
    j = 27643
    for i in y_pred:
        writer.writerow([j, i])
        j += 1
'''
with open('GB20_input05.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Popularity'])
    for i in y_pred:
        writer.writerow([i])
        '''
        