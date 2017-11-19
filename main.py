import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import models as md
import help_functions as hf

import fnmatch
import warnings
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pickle as pkl

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
    return pdd, list(pdd.columns)


train_d, train_cols=get_train_data()

cat_cols=fnmatch.filter(train_cols, '*cat')
train_df=pd.get_dummies(data=train_d, columns=cat_cols)


X_trainset=train_df.drop(["target", "id"], axis=1).as_matrix()
y_trainset=train_df["target"].as_matrix()
xg=md.naive_optimal_xgb_model()
print hf.k_fold_gini(X_trainset, y_trainset, n_splits=3, shuffle=False, random_state=1337, clf_function=xg)



##
# Code for reducing memory. May be improved further
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


