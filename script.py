import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.linear_model import *
import numpy as np
import json

f = open('config.json')
c = json.load(f)
f.close()

df = pd.read_csv("dataset_1.csv")
print('using dataset_1')
df1, df2 = train_test_split(df)

feature_col_list = []
for col in df.columns:
    if 'feature' in col:
        feature_col_list.append(col)

X1 = df1.loc[:, feature_col_list]
y1 = df1.loc[:, 'label']

X2 = df2.loc[:, feature_col_list]
y2 = df2.loc[:, 'label']



for i in range(3):
    model = LogisticRegression(max_iter=500, C=(c['dataset_1']['C'][i]))
    model.fit(X1, y1)
    y3 = model.predict(X2)
    s = accuracy_score(y2, y3)
    print('score for trial ' + str(i) + ' with C=' + str(c['dataset_1']['C'][i]) + ': ' + str(s))


df = pd.read_csv("dataset_2.csv")
print('using dataset_2')
df1, df2 = train_test_split(df)

feature_col_list = []
for col in df.columns:
    if 'variable' in col:
        feature_col_list.append(col)

X1 = df1.loc[:, feature_col_list]
y1 = df1.loc[:, 'label']

X2 = df2.loc[:, feature_col_list]
y2 = df2.loc[:, "label"]



for i in range(4):
    model = LogisticRegression(max_iter=1000, C=c['dataset_2']['C'][i])
    model.fit(X1, y1)
    y3 = model.predict(X2)
    s = accuracy_score(y2, y3)
    print('score for trial ' + str(i) + ' with C=' + str(c['dataset_2']['C'][i]) + ': ' + str(s))


print('coefficients')
for i in range(len(feature_col_list)):
    print(feature_col_list[i], model.coef_[0, i])
