import tflearn

from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv

# getting the data and labels is all the labels like whether you survived or not
# target column 0 because to sort by whether you survived
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2, columns_to_ignore=[2,7])  # columns start at 1

# print("Data: ", data)
# print("Labels", labels)

# female as 1 and male as 0

# should loop through all people (going through larger csv array. Change male and female to binary
for person in data:
    # print(person)
    if person[1] == "male":
        person[1] = 0
    else:
        person[1] = 1

    # print(person)

# Neural Network

# Input Layer
net = tflearn.input_data(shape=[None, 6])  # 6 nodes in the input layer because 6 variables to work with

# Hidden Layers
net = tflearn.fully_connected(net, 32)  # a hidden layer with 32 nodes
net = tflearn.fully_connected(net, 32)  # another hidden layer with 32 nodes

# Output Layer
net = tflearn.fully_connected(net, 2, activation='softmax')  # final has 2 nodes because either survived or not
net = tflearn.regression(net)

# Define Model
model = tflearn.DNN(net)  # DNN Does the training for us

# X-inputs data, Y-targets labels, n_epoch is number of times run. Show metric displays confidence
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# To test the network and print survival rate
dicaprio = [['3',0, '19', '0', '0', '5.000']]  # 3rd class, male, 19y old, no families, $5 ticket
print(model.predict(dicaprio)[0][1] * 100)

winslet = [['1', 1, '17', '1', '2', '150.00']]
print(model.predict(winslet)[0][1] * 100)

smart = [['3', 1, '20', '0', '0', '350.00']]
print(model.predict(smart)[0][1] * 100)