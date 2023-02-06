# DEEPTREE
A simple Decision Tree Classifier

## Overview

This project is a custom implementation of a decision tree classifier inspired from [scikit-learn-decision-tree-classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). The classifier is built from scratch without any external dependencies.

Gini index is used by the classifier to construct the decision tree. Using the gini index for partitioning has allowed the classifier to handle both continuous and categorical features. This allows the classifier to be used in a wider range of applications and datasets.

In addition, the classifier also includes a basic tree printing functionality, which can be used to visualize the tree structure and gain a better understanding of how the classifier is making its decisions.

## Features

* No external dependencies
* Handles both continuous and categorical features
* Uses Gini index to measure impurity of partion
* Includes basic tree printing functionality for tree visualization

# Requirements

Python 3.x
[Optional] Any text editor or IDE of your choice for editing the code.

# Installation

deeptree can be installed using the following command:

```
pip install deeptree
```
or
```
pip3 install deeptree
```

# Dependencies

deeptree is built using only in built python libraries.

## Functionalities

The functions in deeptree package come as part of two classes Node and Classifier. The functions are:-

### deeptree.Node.get_feature_midpoints(i=0)

This function finds the midpoints for any continuous feature corresponding to index 'i' in the dataset.

### deeptree.Node.get_splitting_subsets(i=0)

This function finds the splitting subsets for any discrete feature corresponding to index 'i' in the dataset.

### deeptree.Node.get_gini_value()

This function calculates the gini value of the deeptree node.

### deeptree.Classifier.fit(dataset=[],label_index=-1)

This function trains the decision tree classifier on the given dataset.

### deeptree.Classifier.predict(dataset=[]):

This function predicts the labels/classes of the given dataset.

### deeptree.Classifier.print_tree(node, level=0)
        
This function prints the structure of the decision tree with the details at each node.


Example on how to use deeptree are provided in example.py. The example is based on the [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) dataset downloaded from [UCI achine learning repository](https://archive.ics.uci.edu/ml/index.php).

## License

MIT License

## Acknowledgments

The source code for this project was created as part of one of the courseworks for MSc. Data Science program at Lancaster University. Thanks to Leandro Soriano Marcolino for his amazing classes and this coursework topic.