import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'deeptree'))
from deeptree.decision_tree_classifier import Node, Classifier 


MOCK_DATASET = [[1, 2, 'three', 'label_2'], [4, 5, 'six', 'label_2'], [
    7, 8, 'nine', 'label_2'], [10, 11, 'twelve', 'label_1']]
node = Node(MOCK_DATASET)


@pytest.mark.parametrize('test_input, expected', [(0, [1, 4, 7, 10]), (-1, ['label_2', 'label_2', 'label_2', 'label_1'])])
def test_getIthFeature(test_input, expected):
    assert node._getIthFeatures(test_input) == expected


@pytest.mark.parametrize('test_input, expected', [(0, [2.5, 5.5, 8.5]), (1, [3.5, 6.5, 9.5])])
def test_getFeatureMidPoints(test_input, expected):
    assert node._getFeatureMidPoints(test_input) == expected


def test_getSplittingSubsets():
    output = node._getSplittingSubsets(2)
    features = sorted(node._getIthFeatures(2))
    # checking if all individual subsets are present
    assert sorted([item[0] for item in output[:4]]) == features
    # checking if all 2 member subsets are there
    assert sorted(output[4:], key=lambda item: item[1]) == [
        (output[0][0], feature) for feature in features if feature != output[0][0]]


@pytest.mark.parametrize('test_input, expected', [(None, 0.375)])
def test_getGiniValue(test_input, expected):
    assert node.getGiniValue() == expected


def test_findBestSplit():

    # test case for best split is on a discrete feature
    test_input_1 = [[5, 'high', 0], [10, 'medium', 1], [15, 'low', 0], [5, 'high', 1], [10, 'medium', 0], [
        20, 'low', 1], [20, 'high', 0], [35, 'low', 0], [40, 'low', 1], [50, 'high', 0], [50, 'low', 1]]
    expected_output_1 = (True, 0.4481, [[5, 'high', 0], [5, 'high', 1], [20, 'high', 0], [50, 'high', 0]], [[10, 'medium', 1], [10, 'medium', 0], [
                         15, 'low', 0], [20, 'low', 1], [35, 'low', 0], [40, 'low', 1], [50, 'low', 1]], ('high',), 1)
    # test case for best split is on continuous feature
    test_input_2 = [[5, 'high', 0], [10, 'medium', 1], [15, 'low', 0], [5, 'high', 1], [10, 'medium', 0], [
        20, 'low', 1], [20, 'high', 0], [35, 'low', 0], [40, 'low', 1], [50, 'low', 0], [50, 'low', 1]]
    expected_output_2 = (True, 0.4621, [[5, 'high', 0], [5, 'high', 1], [10, 'medium', 1], [10, 'medium', 0], [15, 'low', 0], [
                         20, 'low', 1], [20, 'high', 0], [35, 'low', 0]], [[40, 'low', 1], [50, 'low', 0], [50, 'low', 1]], 37.5, 0)
    # test case for one data point in dataset
    test_input_3 = [[1, 2, 3, 'yes']]
    # test case when all data points belong to same class/label
    test_input_4 = [[1, 2, 3, 'yes'], [4, 5, 6, 'yes'], [7, 8, 9, 'yes']]
    # test case when all data points have same features
    test_input_5 = [[1, 2, 3, 'yes'], [1, 2, 3, 'no'], [1, 2, 3, 'may_be']]
    expected_output_345 = (False, float('inf'), None, None, None, 0)
    tests = [test_input_1, test_input_2,
             test_input_3, test_input_4, test_input_5]
    results = [expected_output_1, expected_output_2,
               expected_output_345, expected_output_345, expected_output_345]

    for i in range(5):
        node = Node(tests[i])
        dt = Classifier()
        dt.root = node
        dt._setHyperParameters()
        output = dt._findBestSplit(node)
        left, right = node.left, node.right
        if i <= 1:
            left, right = sorted(node.left.data, key=lambda item: item[0]), sorted(
                node.right.data, key=lambda item: item[0])
        assert (output, node.gini, left, right,
                node.threshold, node.feature) == results[i]
