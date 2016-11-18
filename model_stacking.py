import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import csv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X1_tr = pd.read_csv('f_DT6_input02.csv')['Popularity']
X2_tr = pd.read_csv('f_GB20_input02.csv')['Popularity']
X1_te = pd.read_csv('f_DT6_output02.csv')['Popularity']
X2_te = pd.read_csv('f_GB20_output02.csv')['Popularity']

y = pd.read_csv('popularity.csv')
y_tr = y['Popularity']
X_tr = pd.concat([X1_tr, X2_tr], axis=1)
X_te = pd.concat([X1_te, X2_te], axis=1)

pipe = Pipeline([('std', StandardScaler()),
                 ('clf', GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_features='sqrt', subsample=0.8, random_state=0))])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict_proba(X_te)[:,1]
#y_pred = pipe.predict_proba(X_both_one_hot[27643:][:])[:,1]


with open('G_STACK_DT6GB20_GB20output02.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Popularity'])
    j = 27643
    for i in y_pred:
        writer.writerow([j, i])
        j += 1
