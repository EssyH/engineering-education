### Introduction to Ensemble Methods in Machine Learning

### Introduction
One thing I've noticed is that when we try to relate concepts and experiences we encounter in our daily lives, things become easier to learn. When we have a metaphor for real life, it is difficult to remember things. As a result, to keep this in mind, we will examine various methods of comparing ourselves to everyday situations.

The same approaches are used when integrating models in machine learning. To improve overall performance, they combine decisions from multiple models. This can be accomplished in a variety of ways, which are detailed in this article.

The goal of this article is to introduce the concept of learning by combining algorithms and to comprehend the algorithms that use this process. To help you understand this wide range of topics, we will explain step by step the introduction to ensemble methods in machine learning focussing on the problem of real life.


### Table of contents
- [Prerequisites](#prerequisites)
- [Definition of Ensemble Methods](#definition-of-ensemble-methods)
- [Types of Ensemble Methods in Machine Learning](#types-of-ensemble-methods-in-machine-learning)
  - [Homogeneous Ensemble](#homogeneous-ensemble)
  - [Heterogeneous Ensemble](#heterogeneous-ensemble)
  - [Sequential Ensemble Methods](#sequential-ensemble-methods)
  - [Parallel Method](#parallel-method)
- [Technical Classification of Ensemble Methods](#technical-classification-of-ensemble-methods)
- [Similarities between boosting and bagging methods](#similarities-between-bagging-and-boosting-methods)
- [Conclusion](#conclusion)

### Prerequisites
- The reader should have basic knowledge in machine learning algorithms.
- The reader should have read about machine learning life cycles and their advantages.


### Definition of Ensemble Methods
A multimodal system where different classifiers and techniques are grouped together to form a predictive model is referred to as the *ensemble method*. It is compiled as Parallel Model, Heterogeneous, Homogeneous, and Sequential Model methods, etc. 

Ensemble also aids in reducing the variability of predicted data, reducing bias in the predictable model, and accurately differentiating and predicting statistics in complex problems.


### Types of Ensemble Methods in Machine Learning
In machine learning, bias, diversity, and noise all have a negative impact on errors and predictions in machine learning models. A combination of several methods is used to avoid these issues. They include homogeneous ensemble, heterogeneous ensemble, sequential ensemble methods, and parallel ensemble methods.


- ***Homogeneous Ensemble***: A homogeneous ensemble is made up of members who all use the same type of base learning algorithm. Bagging and boosting create diversity by allocating weights to training examples, but the ensemble is typically built using a single type of base classifier.

Bagging and boosting are the most known examples of a homogeneous ensemble.


- ***Heterogeneous Ensemble***: Heterogeneous ensemble is made up of associates with various base learning algorithms, which include, SVM, ANN, and Decision Trees. Stacking, which is analogous to boosting, is a well-known heterogeneous ensemble method.


- ***Sequential Ensemble Methods***: This is a type of ensemble method in which base learners are sequentially produced and where data dependency resides. In this ensemble method, all other data in the base learner relies on previous data in some way. As a result, the previously mislabeled data is tuned based on its weight in order to improve the overall system performance.


- ***Parallel Ensemble Methods***: In this ensemble method, the base learner is produced in parallel sequence, with no data dependency. Every piece of data in the base learner is obtained on its own.



### Technical Classification of Ensemble Methods
Ensemble methods are technically classified as ;

- Boosting
- Bagging
- Random forest
- Stacking

#### (a) Boosting
Boosting ensemble is an ensemble method that constructs a strong classifier from a group of weak classifiers by merging multiple weak classifiers in series. It's an example of sequential ensemble methods whose main aim/property is the concept of correcting prediction errors.

It consists of various algorithms including CatBoost, LPBoost, XGBoost, BrownBoost, AdaBoost, GradientBoost. Among the above algorithms, the first successful boosting algorithm to be designed with the objective of binary classification was the AdaBoost algorithm also known as *the Adaptive Boosting algorithm*. Yoav Freund and Robert Schapire developed it and were later awarded the Gödel Prize in 2003 for the innovation.

>Below is an example of how you implement boosting in python

![Implementing boosting in python](/engineering-education/introduction-to-ensemble-methods-in-machine-learning/boosting.jpg)

#### (b) Bagging
It's one of the parallel ensemble methods that combine numerous versions of a predicted model, it is also known as ***bootstrap aggregating***. Bagging minimizes prediction variance by creating extra data while employing various combinations in the training data.

Generally, bagging works well whereas it serves as the foundation for a wide range of decision tree ensemble algorithms, including the well-known extra trees ensemble algorithms and random forest, as well as Random Subspaces, Pasting, and Random Patches ensemble algorithms.

The Bootstrap sampling method is used to estimate a population statistic from a small sample of data. It is accomplished by generating multiple samples, each statistis sample is computed, and their mean statistic is recorded.

> Below is an example of how bagging is implemented in python

![Implementing bagging in python](/engineering-education/introduction-to-ensemble-methods-in-machine-learning/bagging.jpg)

#### (c) Random forest
It's almost similar to bagging since it also uses deep trees on bootstraps samples. A subset of a sample is chosen at random, hence reducing the likelihood of receiving related prediction values. Its key property is deciding on missing data.

![Random forest](/engineering-education/introduction-to-ensemble-methods-in-machine-learning/random.png)


Random forest is capable of dealing with massive datasets of higher dimensionality thus being one of the dimensionality methods.


>Below is an example of how to implement random forest in python

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
model = RandomForestRegressor(n_estimators = 100, random_state = 0)
model.fit(X_train, y_train)
model.predict(X_test)
```

#### (d) Stacking
The Stacking method is an example of a heterogeneous ensemble method that uses a meta-classifier to combine multiple classifications. Its main aim is to produce a model that is less biased.

Unlike other ensemble methods, its advantage is that it can combine several high-performing models on a  specific classification task to make a prediction that outperforms any single model.

>Here is an example of how to implement stacking in python

![Implementing stacking in python](/engineering-education/introduction-to-ensemble-methods-in-machine-learning/stacking.jpg)

### Similarities between boosting and bagging methods
- ***Stability***: Both bagging and boosting are effective at increasing stability.
- ***Variance***: They are both effective at reducing variance.
- ***Data sets***: In both bagging and boosting multiple training data sets are produced and samples are chosen at random.
- ***Learner***: Both boosting and bagging aim at producing N learners from a single learner.

![Bagging and boosting](/engineering-education/introduction-to-ensemble-methods-in-machine-learning/ense.jpg)

### Conclusion
Ensemble methods have always had a great impact in the machine language world. With the information above the reader is able to tackle issues dealing with ensemble methods. 

You can check out the links below for more articles on ensemble methods ;
- [Ensemble Learning Techniques to Improve Machine Learning](https://www.section.io/engineering-education/ensemble-learning/)
- [Ensemble Learning on Bias and Variance](https://www.section.io/engineering-education/ensemble-bias-var/)
- [Building an Ensemble Learning Based Regression Model using Python](https://www.section.io/engineering-education/ensemble-learning-based-regression-model-using-python/)
- [Saving and Loading Stacked Ensemble Classifiers in ONNX Format in Python](https://www.section.io/engineering-education/save-and-load-stacked-ensembles-in-onnx/)
- [Boosting Algorithms in Python](https://www.section.io/engineering-education/boosting-algorithms-python/)
























