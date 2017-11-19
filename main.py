import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import fnmatch
import warnings
from sklearn.base import clone
from metrics import eval_gini
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pickle as pkl

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
# Returns the normalized_gini coefficient
def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini
##
# K_Fold with respect to normalized_gini
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

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def get_train_data():
    input_file=open("dtypes_train.pkl", 'rb')
    dtypes_train=pkl.load(input_file)
    input_file.close()
    pdd=pd.read_csv('train.csv',
                    delimiter=",",
                    dtype=dtypes_train)
    return pdd
def get_dummies(data, columns):
    return pd.get_dummies(data=data, columns=columns)

# Current top XGB model
def xgb_class_model():
    model=XGBClassifier(n_estimators=400,
                  max_depth=4,
                  objective="binary:logistic",
                  learning_rate=0.07,
                  subsample=0.8,
                  min_child_weight=6,
                  colsample_bytree=0.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3)
    return model


train_d=get_train_data()



train_cols=list(train_d.columns)


cat_cols=fnmatch.filter(train_cols, '*cat')
train_df=get_dummies(data=train_d, columns=cat_cols)


X_trainset=train_df.drop(["target", "id"], axis=1).as_matrix()
y_trainset=train_df["target"].as_matrix()
xg=xgb_class_model()
print k_fold_gini(X_trainset, y_trainset, n_splits=3, shuffle=False, random_state=1337, clf_function=xg)



"""
train_d=pd.read_csv('train.csv', delimiter=",")
columns=list(train_d.columns)
columns.remove('id')
columns.remove('target')
dict_dtypes={}
dict_dtypes["id"]="int64"
dict_dtypes["target"]="int8"

cat_cols=fnmatch.filter(columns, '*cat')
bin_cols=fnmatch.filter(columns, '*bin')
cat_bin_cols=cat_cols+bin_cols
rest_cols=diff(columns, cat_cols+bin_cols)

dict_types={}
for rest in rest_cols:
    dict_types[rest]="float64"
for cat_bin in cat_bin_cols:
    dict_types[cat_bin]="int8"
dict_types["id"]="int64"
dict_types["target"]="int8"

output=open('dtypes_train.pkl', 'wb')
pkl.dump(dict_types, output)
output.close()
"""


