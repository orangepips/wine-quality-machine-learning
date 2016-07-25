'''
Candidate algorithms identified through a combination of Googling on sklearn algorithms and reviewing
WEKA: Data Mining - Practical Machine Learning Tools & Techniques (3rd ed)

OBSERVATIONS

Random Forest provides the best accuracy (<80%) and good runtime.

Gaussian Naive Bayes while less accurate (~70%) is much faster implying better scaling.

'''

import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import time

whitewines = pd.read_csv("winequality-white.csv") # replaced all semicolons with commas & fixed a ",," to "," on line 2729
# bin bad and good wine as (0, 5], (5, 10]
whitewines['is_good'] = pd.cut(whitewines[u'quality'], [0,5,10], labels=[0, 1]) # using "bad", "good" produces an error in sklearn
del whitewines['quality']

data = whitewines[whitewines.columns[:-1]].values
target = whitewines['is_good'].as_matrix()

# http://stackoverflow.com/a/5478448/470838
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

# false positive and false negative rates

@timing
def x_val_score(name, clf, data, target):
    scores = cross_validation.cross_val_score(clf, data, target, cv=10)
    print name + " Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

classifiers = {
    'GaussianNB': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'SVC': svm.SVC(),
    'LinearRegression': linear_model.LinearRegression(),
    'StochasticGradientDescent': SGDClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier()
}

for key in classifiers:
    x_val_score(key, classifiers[key], data, target)

print("FEATURE SELECTION")

for key in classifiers:
    clf = Pipeline([
        ('feature_selection', LinearSVC()),
        ('classification', classifiers[key])
    ])
    x_val_score(key, clf, data, target)

print("OUTLIERS REMOVED")

# remove records with values beyond positive 3rd standard deviation (recall skews were supermajority positive)
pos_third_dev = {}
for column in whitewines.columns[:-1]:
    pos_third_dev[column] = whitewines[column].mean() + whitewines[column].std() * 3

for column in pos_third_dev:
    whitewines = whitewines[whitewines[column] <= pos_third_dev]

for key in classifiers:
    x_val_score(key, classifiers[key], data, target)
