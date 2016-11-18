import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import VotingClassifier
import itertools
from sklearn.svm import SVC

X = pd.read_csv('train_feature_encode.csv')
y = pd.read_csv('popularity.csv')
y = y['Popularity']
X_one_hot = pd.get_dummies(X)

#CVk
for k in [1]:
    pipe = Pipeline([('std', StandardScaler()),
                     ('clf', GradientBoostingClassifier(n_estimators = 20))])

    scores = cross_val_score(estimator=pipe, X=X_one_hot, y=y, cv=5, scoring='roc_auc')
#print(scores.mean(), scores)
    print('[%d-MaxDepth]\nValidation accuracy: %.3f %s' % (k, scores.mean(), scores))
'''
#Voting
pipe1 = Pipeline([('std', StandardScaler()), ('clf', DecisionTreeClassifier(max_depth=6, random_state=0))])
pipe2 = Pipeline([('std', StandardScaler()), ('clf', GradientBoostingClassifier(n_estimators = 20))])


print('[Voting]')
best_vt, best_w, best_score = None, (), -1
for a, b in [[1,1],[1,2],[2,1]]: # try some weight combination
    clf = VotingClassifier(estimators=[('dt', pipe1), ('gb', pipe2)],
                           voting='soft', weights=[a,b])
    scores = cross_val_score(estimator=clf, X=X_one_hot, y=y, cv=5, scoring='roc_auc')
    print('%s: %.3f (+/- %.3f)' % ((a,b), scores.mean(), scores.std()))
    if best_score < scores.mean():
        best_vt, best_w, best_score = clf, (a, b), scores.mean()

print('\nBest %s: %.3f' % (best_w, best_score))



# out of core train
def get_stream(path, size):
    for chunk in pd.read_csv(path, chunksize=size):
        yield chunk

# loss='log' gives logistic regression
clf = SGDClassifier(loss='log', n_iter=100)
batch_size = 1000
stream1 = get_stream(path='train.csv', size=batch_size)
stream2 = get_stream(path='output.csv', size=batch_size)
classes = np.array([0, 1])
train_auc, val_auc = [], []
# we use one batch for training and another for validation in each iteration
iters = int((25000+batch_size-1)/(batch_size*2))
for i in range(iters):
    batch1 = next(stream1)
    batch2 = next(stream2)
    X_train = batch2
    y_train = batch1['Popularity']
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    if X_train is None:
        break
    clf.partial_fit(X_train, y_train, classes=classes)
    train_auc.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
    
    # validate
    batch1 = next(stream1)
    batch2 = next(stream2)
    X_val = batch2
    y_val = batch1['Popularity']
    sc_x = StandardScaler()
    X_val = sc_x.fit_transform(X_val)
    score = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
    val_auc.append(score)
    print('[{}/{}] {}'.format((i+1)*(batch_size*2), 25000, score))
    '''