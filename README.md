Summary
===

Choose the best algorithm by accuracy based on cross validation and folding. For actual test also included relative
weights of classes as argument to the classifier under the assumption the dataset is reflective of actual class
distributions of unseen instances.

RandomForest
White Wine data set
90% training
10% test
False Positive Rate: 0.246913580247
False Negative Rate: 0.109756097561
Accuracy: 0.844897959184

Files
===

 1. explore.py: how data was analyzed
 1. evaluate.py: how algorithms were analyzed
 1. train.py: a solution implementing results of first two

Tools Used
===

 1. WEKA 3.7.1 - used as a sanity check to rapidly iterate through ideas in GUI
 1. WinPython 2.7.6 - basis of python files sent - includes the following packages used
   1. pandas
   1. sklearn
   1. numpy
   1. matplotlib

winequality-white.csv Scrubbing
===

 1. Convert to CSV: replace all ";" with ",".
 1. Line 2729 replace ",," with "," (extra field)

Approach Used
===

 1. Identify instances with attributes whose values may skew learning algorithms (explore.py) - https://onlinecourses.science.psu.edu/stat857/node/223
   1. Outliers: visualize data with histograms and box plots.
   1. Correlation: Pearson and Spearman - values close to |1| imply need for feature removal - http://www3.nd.edu/~mclark19/learn/ML.pdf
 1. Identify possible candidate algorithms and evaluate (evaluate.py):
   1. Dataset
     1. As-is
     1. Feature removal
     1. Outlier removal
   1. Record accuracy & runtime.

