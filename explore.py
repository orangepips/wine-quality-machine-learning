'''
WHAT THIS SCRIPT DOES

Mean average, standard deviation, min, max and quartiles of each column.
Get correlation values for each pair of columns excluding 'quality'.
Visualize data using histograms and box plots.

OBSERVATIONS FROM SCRIPT OUTPUT

Several attributes have significant outliers that certain algorithms are sensitive to - e.g. neural nets.
This can be solved for by removing instances with outlier values from the training set. In this particular case the
super majority are positively skewed, implying only a need to trim in that direction.
    http://www3.nd.edu/~mclark19/learn/ML.pdf &  https://onlinecourses.science.psu.edu/stat857/node/223

Some attributes are strongly correlated |r| > .7 (e.g. e.g. density & residual sugar) which can impact some algorithms -
This can be solved by manually removing attributes or algorithms designed for the purpose
    http://www.biom.uni-freiburg.de/Dateien/PDF/collinearity-a-review-of-methods-to-deal-with-it-and-a-simulation-study-evaluating-and-their-performance

Several attributes appear to have noticeable difference in mean average between "good" and "bad" (e.g. alcohol) in
boxplot charts suggesting linear regression as an approach.
'''
import pandas as pd
import matplotlib.pyplot as plt

whitewines = pd.read_csv("winequality-white.csv") # replaced all semicolons with commas & fixed a ",," to "," on line 2729

# bin bad and good wine as (0, 5], (5, 10]
whitewines['alignment'] = pd.cut(whitewines[u'quality'], [0,5,10], labels=['bad', 'good'])
del whitewines['quality']

print(whitewines.describe())
print(whitewines.corr()) # pearson for linear relationships
print(whitewines.corr(method='spearman')) # monotonic (i.e. non-linear)

whitewines.hist(bins=50, normed=True)

fix, axes = plt.subplots(nrows=3, ncols=4)
row = 0
col = 0
for column in whitewines.columns[:-1]:
    #whitewines[column].plot(ax=axes[row, col], kind='box')
    whitewines.boxplot(column=column, by='alignment', ax=axes[row, col])
    if col == 3:
        row += 1
        col = 0
    else:
        col += 1

plt.show()
