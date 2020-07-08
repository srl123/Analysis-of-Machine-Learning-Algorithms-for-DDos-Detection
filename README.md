# Analysis-of-Machine-Learning-Algorithms-for-DDos-Detection

Used different types of machine learning classifiers such as Passive Aggressive, Extra Trees, Dummy Classifier to detect the DDos attack and compared the accuracies of the classifiers to deterimine the best out of the three. 
I have taken the dataset from this website: 

https://sites.google.com/view/cloudddoddataset/home

Algorithm Description:

# 1. Passive Aggressive Classifier:

•	Passive Aggressive Algorithms are a family of online learning algorithms (for both classification and regression) proposed by Crammer at al.

•	The idea is very simple and their performance has been proofed to be superior to many other alternative methods like Online Perceptron and MIRA.

# Parameters:

class sklearn.linear_model.PassiveAggressiveClassifier(*, C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='hinge', n_jobs=None, random_state=None, warm_start=False, class_weight=None, average=False)

# 2. Extra Trees Classifier:

•	Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result.

•	In concept, it is very similar to a Random Forest Classifier and only differs from it in the manner of construction of the decision trees in the forest.

•	Each Decision Tree in the Extra Trees Forest is constructed from the original training sample.

•	Then, at each test node, each tree is provided with a random sample of k features from the feature-set from which each decision tree must select the best feature to split the data based on some mathematical criteria (typically the Gini Index).

•	This random sample of features leads to the creation of multiple de-correlated decision trees.

# Parameters:

class sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

# 3. Dummy Classifier:

•	A dummy classifier is a type of classifier which does not generate any insight about the data and classifies the given data using only simple rules.

•	The classifier’s behavior is completely independent of the training data as the trends in the training data are completely ignored and instead uses one of the strategies to predict the class label.

•	It is used only as a simple baseline for the other classifiers i.e. any other classifier is expected to perform better on the given dataset.

•	It is especially useful for datasets where are sure of a class imbalance. It is based on the philosophy that any analytic approach for a classification problem should be better than a random guessing approach.

Below are a few strategies used by the dummy classifier to predict a class label –
1.	Most Frequent: The classifier always predicts the most frequent class label in the training data.

2.	Stratified: It generates predictions by respecting the class distribution of the training data. It is different from the “most frequent” strategy as it instead associates a probability with each data point of being the most frequent class label.

3.	Uniform: It generates predictions uniformly at random.

4.	Constant: The classifier always predicts a constant label and is primarily used when classifying non-majority class labels.

# Parameters:

class sklearn.dummy.DummyClassifier(*, strategy='warn', random_state=None, constant=None)


