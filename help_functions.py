import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import clone

##
# Gini coefficient
def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(float(cumPosFound[len(cumPosFound)-1] + df[i][0]))
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)
##
# Normalized Gini coefficient
def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

##
# K-Fold with gini
def k_fold_gini(X_train, y_train,n_splits=2, shuffle=False, random_state=3, clf_function=XGBClassifier()):
    kf=KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    gini_score=[]

    for train_index, test_index in kf.split(X_train):
        clf=clone(clf_function)
        clf.fit(X_train[train_index], y_train[train_index])
        y_pred=clf.predict_proba(X_train[test_index])[:,1]
        gini_score.append(normalized_gini(y_train[test_index], y_pred))
    mean_gini=np.mean(gini_score)
    return "Mean-normalized_gini: %s"%mean_gini

