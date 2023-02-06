# Example on how to use deeptree

import deeptree

dataset = []

# loading iris dataset
with open('./iris.data', 'r') as file:
    while True:
        line = file.readline().strip()
        if not line:
            break
        values = line.lower().split(',')
        for i, value in enumerate(values):
            try:
                values[i] = float(value)
            except:
                pass
            dataset.append(values)


# splitting into test train data

train_data = dataset[10:50] + dataset[60:100] + dataset[110:150]
test_data = dataset[:10] + dataset[50:60] + dataset[100:110]

# The label in iris dataset is present in the last column
label_index = -1

test_features = [item[:label_index] + item[label_index+1:] for item in test_data]
test_labels = [item[label_index] for item in test_data]

# Creating a default classifier
# Can be customised using params
dt = deeptree.Classifier()

# training the decision tree
dt.fit(train_data)

# predicting labels of test data
predictions = dt.predict(test_features)

print(predictions)

# basic visualisation of the decision tree
dt.print_tree()
