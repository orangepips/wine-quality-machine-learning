'''
Training set 90% of the data
Validation set 10% of the data
Training includes passing weighted class ratios to classifier
Show false positive and false negative rates
'''

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

class_column = 'is_good'

whitewines = pd.read_csv("winequality-white.csv")
whitewines[class_column] = pd.cut(whitewines[u'quality'], [0,5,10], labels=[0, 1])
del whitewines['quality']

# http://stackoverflow.com/questions/20082674/unbalanced-classification-using-randomforestclassifier-in-sklearn
counts = whitewines[class_column].value_counts()
positive_weight = float(counts[1]) / float(counts[0])

data = whitewines[whitewines.columns[:-1]].values
target = whitewines[class_column].as_matrix()

# setting "random_state" to constant a makes the results table by always choosing same train and test sets and trees
data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size=.1, random_state=13)
clf = RandomForestClassifier(random_state=13)
sample_weights = np.array([positive_weight if i == 0 else 1 for i in target_train])
clf.fit(data_train, target_train, sample_weight=sample_weights)

predictions = clf.predict(data_test)

# http://en.wikipedia.org/wiki/Receiver_operating_characteristic
negatives = 0 # is_good == 0
positives = 0 # is_good == 0
false_negatives = 0
false_positives = 0
true_negatives = 0
true_positives = 0

for prediction, actual in zip(predictions, target_test):
    if actual == 0:
        negatives += 1
        if prediction == 0:
            true_negatives += 1
        else:
            false_negatives += 1
    else:
        positives += 1
        if prediction == 1:
            true_positives += 1
        else:
            false_positives += 1

'''
print positives
print true_positives
print false_positives

print negatives
print true_negatives
print false_negatives
'''

print("False Positive Rate: " + str(float(false_positives) / float(negatives)))
print("False Negative Rate: " + str(float(false_negatives) / float(positives)))

print("Accuracy: " + str(float(true_positives + true_negatives) / float(positives + negatives)))