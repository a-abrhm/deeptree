from __future__ import annotations
import math
import random
import types
from itertools import combinations


class Node:
    """Class that represents a node of the deeptree decision tree

    Parameters
    ----------
    data : list[list]
        List of samples. Each sample is a list which contains feature
        values and a class/label.
        Features can be continuous or discrete.
        Label is the class to which the sample belongs.
    label_index: int, default = -1
        Index of the label in the sample. To specify the last index,
        give -1.
    feature: int, default = 0
        The index of the feature (in the sample) based on which the
        dataset is split.
    threshold: float | list[tuple]
        The value of split. For continuous features it will be numeric
        and for discrete features
        it will be a subset of all possible values.
    gini: float, default = float('inf')
        Gini index/impurity for the split.
    left and right: Node | None
        Left and right branches to the current node object.

    Attributes
    ----------
    labels: list[list]
        List of lists of which each list is a label and the number of 
        samples of that label.
    n_samples: int
        Number of samples in the dataset.
    label: Any
        The label corresponding to the majority samples in the dataset.

    Methods
    -------
    get_feature_midpoints(i=0)
        Returns the midpoints for any continuous feature in the dataset.
    get_splitting_subsets(i=0)
        Returns the splitting subsets for any discrete feature in
        the dataset.
    get_gini_value()
        Returns the gini value of the deeptree node.
    """

    def __init__(
        self,
        data: list[list] = [],
        label_index: int = -1,
        feature: int = 0,
        threshold: float | list[tuple] = None,
        gini: float = float("inf"),
        left: Node | None = None,
        right: Node | None = None,
    ) -> None:
        self.data = data

        if label_index == -1:
            label_index = len(data[0]) - 1

        self.label_index = label_index
        labs = self._get_ith_features(label_index)
        self.labels = dict.fromkeys(labs, 0)
        for key in labs:
            self.labels[key] += 1

        self.labels = list(self.labels.items())
        self.labels.sort(reverse=True, key=lambda item: item[1])
        self.n_samples = len(data)
        self.label = self.labels[0][0]
        self.feature = feature
        self.threshold = threshold
        self.gini = gini 
        self.left = left 
        self.right = right


    def _get_ith_features(self, i: int=0) -> list:
        """Function to extract the i-th feature from all the samples in
        the dataset
        
        Parameters
        ----------
        i: int, default = 0
            The index of the feature in the samples.

        Returns
        -------
        list[int | float | str]
            List of i-th features.
        """

        return [item[i] for item in self.data]

    def get_feature_midpoints(self, i: int=0) -> list:
        """Function that returns the midpoints of the i-th features
        across the dataset
        
        Parameters
        ----------
        i: int, default = 0
            Index of the feature in the sample.

        Returns
        -------
        list[float]
            List of feature midpoints.
        """

        features = sorted(list(set(self._get_ith_features(i))))
        midpoints = []
        for i in range(len(features) - 1):
            midpoints.append((features[i] + features[i + 1]) / 2)

        return midpoints

    def get_splitting_subsets(self, i: int=0) -> list[tuple]:
        """Function that returns the splitting subsets for the i-th
        discrete feature in the dataset

        Parameters
        ----------
        i: int, default = 0
            The index of the feature in the samples.

        Returns
        -------
        list[tuple]
            List of splitting subsets for the feature.
        """

        features = list(set(self._get_ith_features(i)))
        subset_count = 2 ** (len(features) - 1) - 1
        subsets = []
        for i in range(len(features) // 2):
            new_subsets = list(combinations(features, i + 1))[:subset_count]
            subset_count -= len(new_subsets)
            subsets += new_subsets

        return subsets

    def get_gini_value(self) -> float:
        """Function that calculates the gini value for the dataset
        
        Returns
        -------
        float
            gini index
        """
        
        def sumOfSquaresOfProbabilities(n=0):
            """Recursive function that calculates the sum of squared
            probabilities of all labels in the node.
            
            Parameters
            ----------
            n: int, default = 0
                Number of labels whose probability calculation is left
                to be done.

            Returns
            -------
            float
                Sum of squared probabilities.
            """
            if n == 0:
                return 0
            return (
                self.labels[n - 1][1] / self.n_samples
            ) ** 2 + sumOfSquaresOfProbabilities(n - 1)

        return 1 - sumOfSquaresOfProbabilities(len(self.labels))


class Classifier:
    """Class that represents the deeptree decision tree classifier

    Parameters
    ----------
    max_depth : int | None, defautl = None
        Max depth to which the tree should grow to.
    min_samples_split: int, default = 2
        Minimum number of samples needed at a node inorder to be split.
    min_samples_leaf: int, default = 1
        Minimum number of samples needed to be a leaf node.
    max_features: int | float | {"sqrt", "log2"}, default = None
        Number of features to consider when looking for the best split.
        If int, then used as such.
        If float, it expects a values between 0 and 1 exclusive. Uses 
        that much fraction of the total number of features in the sample.
        If "sqrt" or "log2", then takes the square root or log to base 2 
        of the total number of features in the sample.
        If None, then takes all the features.

    Attributes
    ----------
    root: Node | None
        Root node of the decision tree.
    depth: int
        Actual depth of the decision tree.

    Methods
    -------
    fit(dataset=[],label_index=-1)
        Trains the decision tree on the given dataset.
    predict(dataset=[]):
        Predicts the labels/classes of the given samples.
    print_tree(node, level=0)
        Prints the structure of the decision tree.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None
    ) -> None:
        self.root = None
        self.depth = 0

        self.max_depth = None
        if isinstance(max_depth, int) and max_depth > 0:
            self.max_depth = max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        max_features_dict = {
            "sqrt": lambda x: math.ceil(math.sqrt(x)),
            "log2": lambda x: math.ceil(math.log2(x)),
        }
        if max_features in ("sqrt", "log2"):
            self.max_features = max_features_dict[max_features]
        else:
            self.max_features = max_features

    def _set_hyper_parameters(self) -> None:
        """Function to set the default values to the hyper parameters
        that depend on the values of other parameters"""

        n_samples = self.root.n_samples
        try:
            self.min_samples_split = math.ceil(
                self.min_samples_split
                * (n_samples if 0 < self.min_samples_split < 1 else 1)
            )
            if not 0 < self.min_samples_split < n_samples:
                raise Exception
        except:
            self.min_samples_split = 2

        try:
            self.min_samples_leaf = math.ceil(
                self.min_samples_leaf
                * (n_samples if 0 < self.min_samples_leaf < 1 else 1)
            )
            if not 0 < self.min_samples_leaf < n_samples:
                raise Exception
        except:
            self.min_samples_leaf = 1

        try:
            if isinstance(self.max_features, types.FunctionType):
                self.max_features = self.max_features(len(self.root.data[0]) - 1)
            else:
                self.max_features = math.ceil(
                    self.max_features
                    * ((len(self.root.data[0]) - 1) if 0 < self.max_features < 1 else 1)
                )
                if not 0 < self.max_features <= (len(self.root.data[0]) - 1):
                    raise Exception
        except:
            self.max_features = len(self.root.data[0]) - 1


    def _find_best_split(self, node: Node) -> bool:
        """Function that finds the best split for the node based on the
        gini partiion value. Based on the best split, the corresponding
        gini, threshold, feature index and left and right node are also
        found and assigned to the class attributes.
        
        Parameters
        ----------
        node: Node
            Node of the decision tree for which the split is calculated.

        Returns
        -------
        bool
            Returns True or False based on if a successful split is
            found or not.
        """

        # if all the items in the node belong to the same label,
        # then it is not possible to split the node
        # Also if there is only one item in the node, then it cannot be
        # split
        if 1 in [node.n_samples, len(node.labels)]:
            return False

        # looping through features
        # if max_features is less than the total features, then they are
        # randomly selected
        feature_indices = list(range(len(node.data[0]) - 1))
        random.shuffle(feature_indices)
        for idx, i in enumerate(feature_indices):
            # Only taking random max_features from total features
            # if a valid split has not been found within this, then the
            # loop continues till a split is found
            if idx >= self.max_features and node.left is not None:
                break
            # check if the feature is discrete or continuous
            if isinstance(node.data[0][i], str):
                subsets = node.get_splitting_subsets(i)
                # if all the i-th features are same, then there are no
                # subsets and the dataset cannot be split based on the
                # i-th feature
                if len(subsets) == 0:
                    continue
                # looping through splitting subsets
                for subset in subsets:
                    d1, d2 = (
                        [],
                        [],
                    )
                    # variables to store left and right data sets after
                    # splitting
                    for item in node.data:  # looping through data set
                        if (
                            item[i] in subset
                        ):  # splitting by checking membership to subset
                            d1.append(item)
                        else:
                            d2.append(item)

                    # creating left and right nodes from d1 and d2
                    node1, node2 = Node(d1, node.label_index), Node(d2,
                    node.label_index)

                    # calculating the gini partition value for the i-th
                    # feature and current split
                    gini_value = (
                        node1.n_samples / node.n_samples
                    ) * node1.get_gini_value() + (
                        node2.n_samples / node.n_samples
                    ) * node2.get_gini_value()
                    if (
                        gini_value < node.gini
                    ):
                    # updating if the new gini value is the least 
                    # till now
                        node.gini = gini_value
                        node.left, node.right = node1, node2
                        node.threshold = subset
                        node.feature = i
            else:
                # continuous feature
                # sorting the dataset by i-th feature values
                node.data = sorted(node.data, key=lambda item: item[i])
                midpoints = node.get_feature_midpoints(
                    i
                )  # midpoints of i-th feature list
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
                    node1, node2 = Node(d1, node.label_index), Node(d2,
                    node.label_index)

                    # calculating the gini partition value for the i-th
                    # feature and current midpoint
                    gini_value = (
                        node1.n_samples / node.n_samples
                    ) * node1.get_gini_value() + (
                        node2.n_samples / node.n_samples
                    ) * node2.get_gini_value()
                    if (
                        gini_value < node.gini
                    ):
                    # updating if the new gini value is the least till now
                        node.gini = gini_value
                        node.left, node.right = node1, node2
                        node.threshold = midpoint
                        node.feature = i

        # if the dataset cannot be split by none of the features, then
        # false is returned
        if node.left == None:
            return False
        node.gini = round(node.gini, 4)
        return True

    def fit(self, dataset: list[list]=[], label_index = -1) -> None:
        """Function that trains the decision tree based on the given
        dataset
        
        Parameters
        ----------
        dataset: list[list]
            List of samples. Each sample is a list which contains
            feature values and a class/label.
            Features can be continuous or discrete.
            Label is the class to which the sample belongs.
        label_index: int, default = -1
            Index of the label in the sample. To specify the last index,
            give -1.
        """

        if (
            type(dataset) is not list
            or not len(dataset)
            or type(dataset[0]) is not list
            or not len(dataset[0])
        ):
            raise Exception("Invalid input for dataset!")

        # start tree creation
        node = Node(dataset, label_index)
        self.root = node
        # setting hyper parameters of the decision tree classifier
        self._set_hyper_parameters() 
        self._fit(node)

    def _fit(self, node: Node=None, depth: int=0) -> None:
        """Recursive function that continues the tree building
        
        Parameters
        ----------
        node: Node
            Node to be split and continue the tree creation.
        depth: int, default: 0
            Current depth of the decision tree or level of the
            node from the root node.
        """

        # updating the max_depth of the tree
        if depth > self.depth:
            self.depth = depth

        # continue tree creation if max depth hasn't reached
        if self.max_depth is None or self.max_depth > depth:
            # continue tree creation if there are sufficient number of
            # samples to split
            if node.n_samples >= self.min_samples_split:
                output = self._find_best_split(node)

                if output:  # if not leaf node, then continue with the next split
                    # continue with splitting if both left and right
                    # branches will receive sufficient samples
                    if (
                        node.left.n_samples >= self.min_samples_leaf
                        and node.right.n_samples >= self.min_samples_leaf
                    ):
                        self._fit(node.left, depth + 1)
                        self._fit(node.right, depth + 1)

    def _predict(self, node: Node, data: list=[]) -> int | float | str:
        """Function that predicts the label/class of a given sample
        
        Parameters
        ----------
        node: Node
            Root node of the decision tree.
        data: list
            A list of features corresponding to a sample.

        Returns
        -------
        int | float| str
            Predicted label/class of the sample.
        """
        
        if (
            node.left is None
        ):  # when we reach the leaf node, return the classifier output
            return node.label

        # if the feature is discrete then check membership
        if isinstance(node.threshold, tuple):
            if data[node.feature] in node.threshold:
                return self._predict(node.left, data)
            else:
                return self._predict(node.right, data)
        # if feature is continuous then its value is compared against
        # midpoint value
        elif data[node.feature] <= node.threshold:
            return self._predict(node.left, data)
        else:
            return self._predict(node.right, data)

    def predict(self, dataset: list[list]=[]) -> list:
        """Function that predicts the labels/classes of the given
        samples.
        
        Parameters
        ----------
        dataset: list[list]
            List of lists. Each inner list is a sample (set of features
            without labels).

        Returns
        -------
        list
            List of predicted labels.
        """

        predicted_labels = []
        for data in dataset:
            predicted_labels.append(self._predict(self.root, data))

        return predicted_labels

    def print_tree(self, node: Node=None, level: int=0) -> None:
        """Recursive function that prints the decision tree structure with
        details of each node
        
        Parameters
        ----------
        node: Node
            Current node being printed.
        level: int, default = 0
            Level of the node being printed.
        """

        if level == 0:
            node = self.root

        if node:
            print("     " * level, f"level = {level}")
            if node.threshold:
                preposition = "in" if isinstance(node.threshold, tuple) else "<="
                print(
                    "     " * level,
                    f"feature_{node.feature + 1} {preposition} {node.threshold}",
                )
                print("     " * level, f"samples = {node.n_samples}")
                print("     " * level, f"gini = {node.gini}")

            print("     " * level, f"labels = {node.labels}")
            print("     " * level, f"class = {node.label}")

            if node.left:
                print()
                print("     " * (level + 1), "left branch")
                self.print_tree(node.left, level + 1)
            if node.right:
                print()
                print("     " * (level + 1), "right branch")
                self.print_tree(node.right, level + 1)
