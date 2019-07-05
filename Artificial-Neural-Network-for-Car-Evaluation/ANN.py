import numpy as np
import pandas as pd
import random

# read processed data
def read(path):
    data = pd.read_csv(path, header=0)
    return np.array(data)
  #TEST OK

# randomly split data into training data and testing data
def split(data, percent):
    ratio = float(percent) / 100
    train = []
    test = []
    for i in range(len(data)):
        if(random.random() < ratio):
            train.append(data[i])
        else :
            test.append(data[i])
    return np.array(train), np.array(test)
    # TEST OK, train have percent% of original data
    
# initialize all weights
def initWeis(n_hidden, n_neurons, nattri):
    # store the number of neurons in each layer(including hidden and output layer)
    nnodes = []
    for i in range(n_hidden):
        nnodes.append(n_neurons)
    nnodes.append(1)
    weights = []
    num_cur = nattri
    for i in range(len(nnodes)):
        num_next = nnodes[i]
        weights.append(init(num_cur, num_next))
        num_cur = num_next
    return weights

# initialize weights between layer num_cur and num_next
def init(num_cur, num_next):
    return np.random.randn(num_cur, num_next)

# forward pass
def forward(example, weights):
    data = np.array(example)
    data = np.delete(data, len(data) - 1)
    outputs = []
    outputs.append(data)
    for i in range(len(weights)):
        data = np.dot(data, weights[i])
        data = sigmoid(data)
        outputs.append(data)
    return outputs

def sigmoid(x):
    return np.array(1 / (1 + np.exp(x)))
    
# backward pass
def backward(outputs, t, weights, learning_rate):
    n = len(outputs)
    deltas = []
    for i in range(n - 1, -1, -1):
        o = outputs[i]
        if i == n - 1: # output layer
            delta = np.multiply(o, 1 - o)
            delta = np.multiply(delta, t - o)
            deltas.insert(0, delta)
        else:
            delta = deltas[0]
            twod_o = o.reshape(o.shape[0], -1) 
            # print(o)
            # print(twod_o) o and twod_o is diffrent[ 0.89361301  0.90544851], [[ 0.89361301] [ 0.90544851]]
            #print(delta)
            deltaw = np.multiply(learning_rate, np.multiply(twod_o, delta))
            #print(deltaw)
            newdelta = np.multiply(o, 1-o)
            updateddelta = np.dot(delta, weights[i].transpose())
            newdelta = np.multiply(newdelta, updateddelta)
            deltas.insert(0, newdelta)
            
            weights[i] = weights[i] + deltaw
    return weights
    
def backpropagation(train, learning_rate, n_iteration, n_hidden, n_neurons):
    nrow = train.shape[0]
    ncol = train.shape[1]
    # print("number of rows in the data: ", nrow)
#     print("number of cols in the data: ", ncol)
    weights = initWeis(n_hidden, n_neurons, ncol - 1)
    # print("get weights shape")
#     print(weights)
    for i in range(nrow):
        if(i >= n_iteration):
            return weights
        outputs = forward(train[i], weights)
        # print("get outputs")
#         print(outputs)
        backward(outputs, train[i, ncol - 1], weights, learning_rate)
    return weights

def accuracy(test, weights):
    nrow = test.shape[0]
    ncol = test.shape[1]
    correct = 0
    wrong = 0
    for i in range(nrow):
        outs = []
        outs = forward(test[i], weights)
        o = outs[len(outs) - 1]
        t = test[i][ncol - 1]
        if(int(o) == t):
            correct += 1
        else:
            wrong += 1
    return 100 * float(correct) / nrow

path = input("Please enter your processed data path: ")
learn_rate = eval(input("Please give the learning rate, it should between 0~1: "))
n_iterations = eval(input("Please enter the maximum number of iterations: "))
n_hidden = eval(input("Please enter the number of hidden layers: "))
n_neurons = eval(input("Please enter the number of neurons in hidden layer: "))
percent = eval(input("Please enter the percentage of data that used for training(1~99): "))
data = read(path)
train, test = split(data, percent)
weights = backpropagation(train, learn_rate , n_iterations, n_hidden, n_neurons)
print()
print("*************************Print results***********************")
for i in range(len(weights)):
    if(i == 0):
        s = "(Input Layer):"
    elif(i == len(weights) - 1):
        s = "(Last hidden layer):"
    else:
        s = "(" + str(i) + "th hidden layer):"
    print("Layer ", i, s)
    for j in range(len(weights[i])):
        ss = weights[i][j]
        print("         Neuron ", j, "weights: ", ss)
        print()

acc = accuracy(test, weights)
print("Accuracy on testing data is ", acc)