import math
import random
import types
from itertools import combinations


class Node:
    # class that defines the node of a decision tree
    def __init__(self, dataset=[], label_index = -1,feature = 0, threshold = None, gini = float('inf'), left = None, right = None) -> None:
        
        self.data = dataset  # dataset
        # the dataset is a list of data points each of which can have one or more
        # features (continuous or discrete) and the last value will be the label
        # to which it belongs

        # label_index is used to specify which index/column is the label/class
        # by default it takes the last column
        if label_index == -1:
            label_index = len(dataset[0]) - 1

        # to get the label values for each data points
        labs = self._getIthFeatures(label_index)
        self.labels = dict.fromkeys(labs, 0)  # creating a dictionary of labels
        for key in labs:  # looping through labels
            self.labels[key] += 1  # keeping a count of each label
        # storing the label-count values as a list
        self.labels = list(self.labels.items())
        self.labels.sort(reverse=True, key=lambda item: item[1])

        self.n_samples = len(dataset)

        # class/label corresponding to majority of samples
        self.label = self.labels[0][0]

        self.feature = feature  # the feature using which the split is made
        self.threshold = threshold  # value of the feature using which the split is done
        # for continuous features threshold will be numeric values, where as
        # for discrete features it will be a splitting subset of values
        
        self.gini = gini  # gini index / impurity for the split at this node
        self.left = left  # left node of the tree
        self.right = right  # right node of the tree


    # function to extract the i-th feature across all data points
    def _getIthFeatures(self, i=0):
        return [item[i] for item in self.data]

    # function that returns the midpoints for i-th feature across the dataset
    def _getFeatureMidPoints(self, i=0):
        # using set() to remove duplicate feature values
        features = sorted(list(set(self._getIthFeatures(i))))
        midpoints = []  # variable to store the feature midpoints and corresponding split index
        # findding the midpoints of the sorted feature list
        for i in range(len(features) - 1):
            midpoints.append((features[i] + features[i+1]) / 2)

        return midpoints

    # function that returns the splitting subsets for features with discrete values
    def _getSplittingSubsets(self, i=0):
        features = list(set(self._getIthFeatures(i)))
        subset_count = 2 ** (len(features) - 1) - 1
        subsets = []
        for i in range(len(features) // 2):
            new_subsets = list(combinations(features, i + 1))[:subset_count]
            subset_count -= len(new_subsets)
            subsets += new_subsets

        return subsets


    # function that returns the gini value of a decision tree node
    def getGiniValue(self):
        # function that calculates the sum of squared probabilities of each label in the node
        def sumOfSquaresOfProbabilities(n=0):
            if n == 0:
                return 0
            return (self.labels[n - 1][1] / self.n_samples) ** 2 + sumOfSquaresOfProbabilities(n - 1)
        # gini value is 1 - sum of squared probabilities
        return 1 - sumOfSquaresOfProbabilities(len(self.labels))



class Classifier:
    # class that defines a decision tree classifier and its methods

    def __init__(self,max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None) -> None:  # constructor        

        self.root = None # root node of decision tree
        self.depth = 0 # depth of decision tree

        # max depth the tree should grow to
        self.max_depth = None
        if isinstance(max_depth, int) and max_depth > 0:
            self.max_depth = max_depth

        # minimum number of samples needed at a node inorder to be split
        self.min_samples_split = min_samples_split

        # minimum number of samples needed to be a leaf node
        self.min_samples_leaf = min_samples_leaf

        # possible values of max_features
        max_features_dict = {"sqrt": lambda x: math.ceil(
            math.sqrt(x)), "log2": lambda x: math.ceil(math.log2(x))}
        # max_features is used to control the number of features considered in finding the best split
        if max_features in ("sqrt", "log2"):
            self.max_features = max_features_dict[max_features]
        else:
            self.max_features = max_features


    # function to set the default values to the hyper parameters that depend on the values of other parameters
    def _setHyperParameters(self):
        n_samples = self.root.n_samples
        try:
            self.min_samples_split = math.ceil(
                self.min_samples_split * (n_samples if 0 < self.min_samples_split < 1 else 1))
            if not 0 < self.min_samples_split < n_samples:
                raise Exception
        except:
            self.min_samples_split = 2

        try:
            self.min_samples_leaf = math.ceil(
                self.min_samples_leaf * (n_samples if 0 < self.min_samples_leaf < 1 else 1))
            if not 0 < self.min_samples_leaf < n_samples:
                raise Exception
        except:
            self.min_samples_leaf = 1

        try:
            if isinstance(self.max_features, types.FunctionType):
                self.max_features = self.max_features(len(self.root.data[0]) - 1)
            else:
                self.max_features = math.ceil(
                    self.max_features * ((len(self.root.data[0]) - 1) if 0 < self.max_features < 1 else 1))
                if not 0 < self.max_features <= (len(self.root.data[0]) - 1):
                    raise Exception
        except:
            self.max_features = len(self.root.data[0]) - 1
    

    # function that finds the best split for the node based on the gini partiion value
    def _findBestSplit(self, node):
        # if all the items in the node belong to the same label,
        # then it is not possible to split the node
        # Also if there is only one item in the node, then it cannot be split
        if 1 in [node.n_samples, len(node.labels)]:
            return False

        # looping through features
        # if max_features is less than the total features, then they are randomly selected
        feature_indices = list(range(len(node.data[0]) - 1))
        random.shuffle(feature_indices)
        for idx, i in enumerate(feature_indices):
            # Only taking random max_features from total features
            # if a valid split has not been found within this, then the loop
            # continues till a split is found
            if idx >= self.max_features and node.left is not None:
                break
            # check if the feature is discrete or continuous
            if isinstance(node.data[0][i], str):
                subsets = node._getSplittingSubsets(i)
                # if all the i-th features are same, then there are no subsets
                # and the dataset cannot be split based on the i-th feature
                if len(subsets) == 0:
                    continue
                # looping through splitting subsets
                for subset in subsets:
                    d1, d2 = [], []  # variables to store left and right data sets after splitting
                    for item in node.data:  # looping through data set
                        if item[i] in subset:  # splitting by checking membership to subset
                            d1.append(item)
                        else:
                            d2.append(item)

                    # creating left and right nodes from d1 and d2
                    node1, node2 = Node(d1), Node(d2)

                    # calculating the gini partition value for the i-th feature and current split
                    gini_value = (node1.n_samples / node.n_samples) * node1.getGiniValue() + (node2.n_samples / node.n_samples) * node2.getGiniValue()
                    if gini_value < node.gini:  # updating if the new gini value is the least till now
                        node.gini = gini_value
                        node.left, node.right = node1, node2
                        node.threshold = subset
                        node.feature = i
            else:
                # continuous feature
                # sorting the dataset by i-th feature values
                node.data = sorted(node.data, key=lambda item: item[i])
                midpoints = node._getFeatureMidPoints(
                    i)  # midpoints of i-th feature list
                # if all the i-th features are same, then there are no midpoints
                # and the dataset cannot be split based on the i-th feature
                if len(midpoints) == 0:
                    continue
                # looping through midpoints
                for midpoint in midpoints:
                    # splitting the data set at the midpoint
                    # d1 and d2 will store the left and right data sets after splitting
                    d1, d2 = [], []
                    for item in node.data:
                        if item[i] <= midpoint:
                            d1.append(item)
                        else:
                            d2.append(item)

                    # creating left and right nodes from d1 and d2
                    node1, node2 = Node(d1), Node(d2)

                    # calculating the gini partition value for the i-th feature and current midpoint
                    # calculating the gini partition value for the i-th feature and current split
                    gini_value = (node1.n_samples / node.n_samples) * node1.getGiniValue() + (node2.n_samples / node.n_samples) * node2.getGiniValue()
                    if gini_value < node.gini:  # updating if the new gini value is the least till now
                        node.gini = gini_value
                        node.left, node.right = node1, node2
                        node.threshold = midpoint
                        node.feature = i

        # if the dataset cannot be split by none of the features, then false is returned
        if node.left == None:
            return False
        node.gini = round(node.gini, 4)
        return True

    # function that will train the decision tree based on the inputs given
    def fit(self, dataset = []):
        # dataset: the training dataset which consists of features and corresponding label
        # dataset is expected to be a list of lists, with each list containing the feature values
        # and the last column being the label/class

        if type(dataset) is not list or not len(dataset) or type(dataset[0]) is not list or not len(dataset[0]):
            raise Exception("Invalid input for dataset!")

        # start tree creation
        node = Node(dataset)
        self.root = node
        self._setHyperParameters() # setting hyper parameters of the decision tree classifier
        self._fit(node)

    # function that will train the decision tree based on the inputs given
    def _fit(self, node = None, depth = 0):
        # node: decision tree node
        # depth: the depth of the current node in the decision tree

        self.depth = depth # storing the max_depth of the tree

        # continue tree creation if max depth hasn't reached
        if self.max_depth is None or self.max_depth > depth:
            # continue tree creation if there are sufficient number of samples to split
            if node.n_samples >= self.min_samples_split:
                output = self._findBestSplit(node)

                if output:  # if not leaf node, then continue with the next split

                    # continue with splitting if both left and right branches will receive sufficient samples
                    if node.left.n_samples >= self.min_samples_leaf and node.right.n_samples >= self.min_samples_leaf:
                        self._fit(node.left, depth+1)
                        self._fit(node.right, depth+1)


    # function that will predict the class of a single data point
    def _predict(self, node, data):
      # data: a list of feature values of the data sample to be classified
        if node.left is None:  # when we reach the leaf node, return the classifier output
            return node.label

        # if the feature is discrete then check membership
        if isinstance(node.threshold, tuple):
            if data[node.feature] in node.threshold:
                return self._predict(node.left, data)
            else:
                return self._predict(node.right, data)
        # if feature is continuous then its value is compared against midpoint value
        elif data[node.feature] <= node.threshold:
            return self._predict(node.left, data)
        else:
            return self._predict(node.right, data)

    # function that will predict the classes of the given set of data points
    def predict(self, dataset=[]):
        # dataset: a list of data points each of which is a list of feature values
        predicted_labels = []
        for data in dataset:
            predicted_labels.append(self._predict(self.root, data))

        return predicted_labels

    # function that will print the decision tree structure with
    # details at each level
    def printTree(self, node = None, level = 0):
        if level == 0:
            node = self.root
        
        if node:
            print("     " * level, f"level = {level}")
            if node.threshold:
                preposition = 'in' if isinstance(node.threshold, tuple) else '<='
                print("     " * level,
                    f"feature_{node.feature + 1} {preposition} {node.threshold}")
                print("     " * level, f"samples = {node.n_samples}")
                print("     " * level, f"gini = {node.gini}")

            print("     " * level, f"labels = {node.labels}")
            print("     " * level, f"class = {node.label}")

            if node.left:
                print()
                print("     " * (level + 1), "left branch")
                self.printTree(node.left, level + 1)
            if node.right:
                print()
                print("     " * (level + 1), "right branch")
                self.printTree(node.right, level + 1)
        