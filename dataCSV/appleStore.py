import tflearn

from tflearn.data_utils import load_csv

# getting the data and labels is all the labels like whether you survived or not
# target column 0 because to sort by whether you survived
# data, labels = load_csv('AppleStore.csv', target_column=3, categorical_labels=True, n_classes=1, columns_to_ignore=[1,2,4,7,9,10,11,12,13,14,15,16])
data, labels = load_csv('AppleStore.csv', target_column=5, categorical_labels=True, n_classes=0, columns_to_ignore=[1,2,4,7,9,10,11,12,13,14,15,16])

# target column is stuck as a long?

# print("Data: ", data)
# print("Labels", labels)

# Size(bytes), Price, Total Ratings, User Rating

# should loop through all people (going through larger csv array. Change male and female to binary
for app in data:
    print(app)

# Neural Network

# Input Layer
net = tflearn.input_data(shape=[None, 4])  # 4 nodes in the input layer because 4 variables to work with

# Hidden Layers (as many or little as want)
net = tflearn.fully_connected(net, 32)  # a hidden layer with 32 nodes
net = tflearn.fully_connected(net, 32)  # another hidden layer with 32 nodes

# Output Layer
net = tflearn.fully_connected(net, 11098, activation='softmax')  # final has 1 node as price
net = tflearn.regression(net)

# Define Model
model = tflearn.DNN(net)  # DNN Does the training for us
#
# X-inputs data, Y-targets labels, n_epoch is number of times run. Show metric displays confidence
model.fit(data, labels, n_epoch=3, batch_size=16, show_metric=True)


# To test the network


