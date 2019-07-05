import numpy as np
import csv as csv
import math
import random

# This code contains the following functions
# 1. read data from CSV file, get header and data(include attribute value and label)
# 2. calculate information gain, Build Tree for the model, return root node of the tree
# 3. represent the tree, noted as id3_tree
# 4. pruning the tree, noted as prune_tree
# 5. Build tree using random attributes, noted as prune_tree
# 6. predict class for each instance in validation data, test data on the is3_tree model, 
#    prune_tree model, random_tree model , calculate accuracy for each model

# read dataset from input
def readData(path):
    inputdata = csv.reader(open(path, 'r'))
    data = []
    
    for row in inputdata:
        data.append(row)
    # get the attribute name from the 0th row in data, remove 0th row in data
    header = data[0]
    data.pop(0)
    return header, data
    # TEST head and data, dd = readData('training_set.csv') print(head) OK

# Calculate information gain
def calCounts(data):
    if (not data or len(data) == 0):
        return [[0,0,0,0]]
    nrow = len(data)
    ncol = len(data[0])
    # get the number of (x,y), x is the attribute value, y is label, represent the number of (T,T),(T,F),(F,T),(F,F)
    counts = [[0,0,0,0] for i in range(ncol)]
    
    for col in range(ncol):
        for row in range(nrow):
            val = data[row][col]
            if (val == '1'):
                if(data[row][-1] == '1'): counts[col][0] += 1
                else: counts[col][1] += 1
            else:
                if(data[row][-1] == '1'): counts[col][2] += 1
                else: counts[col][3] += 1
    return counts
    # TEST OK head, data = readData('training_set.csv') counts = calCounts(data) print(counts)

def calEntropy(Tnum, Fnum):
    total = Tnum + Fnum
    if(Tnum == 0 or Fnum == 0):
        return 0
    probT = float(Tnum) / total
    probF = float(Fnum) / total
    return - probT * math.log(probT, 2) - probF * math.log(probF, 2)
    
# Get the index of attribute to split, which is the maximum information gain, minimum children entropy
def splitIndex(counts):
    ncol = len(counts)
    index = 0
    min = 1
    
    for i in range(ncol - 1):
       kidsEn = calH(counts, i)
       if(kidsEn <= min):
            min = kidsEn
            index = i
    return index
    # TEST OK head, data = readData('training_set.csv') print(splitIndex(data))

def calH(counts, i ):
    tt = counts[i][0]
    tf = counts[i][1]
    ft = counts[i][2]
    ff = counts[i][3]
    h0 = calEntropy(ft, ff)
    h1 = calEntropy(tt, tf)
    total = counts[i][0] + counts[i][1] + counts[i][2] + counts[i][3]
    H = (float(ft + ff) / total) * h0 + (float(tt + tf) / total) * h1
    return H

# Divide the parent data to two parts, attrval == 0 to left part, attrval == 1 to right part
def divData(data, splitIndex):
    leftData = []
    rightData = []
    for i in range(len(data)):
        row = data[i]
        val = row[splitIndex]
        if (val == '0'):
            leftData.append(row)   
        else:
            rightData.append(row)
    return leftData, rightData
    #TEST OK head, data = readData('training_set.csv') left, right = divData(data) print(len(data)) print(len(left)) print(len(right)) print(left)

# Define TreenNode structure   
class TreeNode(object):
    def __init__(self, attrName, attrVal, nodeData, left, right, level, output, label):
        self.attrName = attrName
        self.attrVal = attrVal
        self.nodeData = nodeData
        self.left = left
        self.right = right
        self.level = level
        self.output = output
        self.label = label
    
    def printFormat(self):
        if (self.level == 0):
            return ""
        s = ''
        if (self.level > 0):
            for i in range(self.level - 1):
                s = s + ' | '
        s = s + self.attrName + ' = ' + self.attrVal + ' : '
        if (self.left == None and self.right == None):
            s = s + self.output
        return s
    
# Build Tree based on ID3 algorithm
def buildTree(data, header, level, attrName, attrVal):
    #determine current data is leaf or not
    counts = calCounts(data)
    setclass = counts[len(counts) - 1]
    dataEntropy = calEntropy(setclass[0]+ setclass[2], setclass[1]+setclass[3])
    # if data instances are identical, case 1. different class cause no attribute to split, return majority value as output 
    # case 2. same class, this case is as same as setEntropy is 0, pure data->leaf
    if (dataEntropy == 0 or identical(data)):
        if (setclass[0] > setclass[3]):
            output = '1'
        else: output = '0'
        return TreeNode(attrName, attrVal, data, None, None, level, output, None) 
 
    # if not leaf, the data can be split to left and right subtree
    index = splitIndex(counts) 
    # get dataset for left_child and right_child
    leftdata, rightdata = divData(data, index)
    leftchild = buildTree(leftdata, header, level + 1, header[index], "0")
    rightchild = buildTree(rightdata, header, level + 1, header[index], "1")
    node = TreeNode(attrName, attrVal, data, leftchild, rightchild, level, None, None)
    return node

# Build tree by randomly picking attributes for each node
def randomBuildTree(data, header, level, attrName, attrVal, selectedlist):
    counts = calCounts(data)
    setclass = counts[len(counts) - 1]
    dataEntropy = calEntropy(setclass[0]+ setclass[2], setclass[1]+setclass[3])
    if (dataEntropy == 0 or identical(data)):
        if (setclass[0] > setclass[3]):
            output = '1'
        else: output = '0'
        return TreeNode(attrName, attrVal, data, None, None, level, output, None) 
 
    randomIndex, list = randomsplit(header, selectedlist)
    leftdata, rightdata = divData(data, randomIndex)
    leftchild = randomBuildTree(leftdata, header, level + 1, header[randomIndex], "0", list)
    rightchild = randomBuildTree(rightdata, header, level + 1, header[randomIndex], "1", list)
    node = TreeNode(attrName, attrVal, data, leftchild, rightchild, level, None, None)
    return node

# Get the index of attribute to split and current selected list
def randomsplit(header, selectedlist):
    indexlist = random.sample(range(len(header) - 1), 10)
    for index in indexlist:
        if(not contains(selectedlist, index)):
            selectedlist.append(index)
            return index, selectedlist

# return true if list contains the element index    
def contains(list, index):
    for i in range(len(list)):
        if(list[i: i + 1] == index):
            return True
    return False

# determine whether the data has same instance(no need to consider their class)
def identical(data):
    lastrow =  data[len(data) - 1]
    for col in range(len(data[0]) - 1):
        for row in range(len(data) - 1):
            if(data[row][col] != lastrow[col]):
                return False
    return True

# get the average depth for the decision tree
def avgdepth(root):
    totalnodes, totalleaf = calnumber(root)
    sumdep = sumdepth(root)
    return float(sumdep)/totalleaf

# get sum of depth of the leaf nodes
def sumdepth(node):
    if (node == None):
        return 0
    if (node.left == None and node.right == None):
        return node.level
    leftsum = sumdepth(node.left)
    rightsum = sumdepth(node.right)
    return leftsum + rightsum
                    
# label each node, root is labeled 1, root.left is labeled 2 and so on.
def labelTree(root):
    label = 1
    queue = [root]
    while (len(queue) != 0):
        size = len(queue)
        for i in range(size):
            cur = queue[0]
            cur.label = label
            if (cur.left != None): queue.append(cur.left)
            if (cur.right != None): queue.append(cur.right)
            del queue[0]
            label += 1
    return

# create a mapping for attribute name and attribute index
def mapping(header):
    dict = {}
    for i in range(len(header)):
        attrname = header[i]
        dict[attrname] = i
    return dict
    # TEST OK head, data = readData('training_set.csv') print(mapping(head))

# predict the class(0 or 1) for a given instance
def predictClass(root, record, dict):
    node = root
    while (node.output == None):
        attrVal = node.left.attrVal
        attrName = node.left.attrName
        index = dict[attrName]
        if (record[index] == attrVal):
            node = node.left
        else:
            node = node.right
    return node.output
    
# calculate total number of nodes and total number of leafnodes in the tree
def calnumber(node):
    if (node.left == None and node.right == None):
        return 1, 1
    leftTotal, leftleaf = calnumber(node.left)
    rightTotal, rightLeaf = calnumber(node.right)
    return leftTotal + rightTotal + 1, leftleaf + rightLeaf

# Copy a decision tree
def copyTree(root):
    if(root == None):
        return None
    left = copyTree(root.left)
    right = copyTree(root.right)
    node = TreeNode(root.attrName, root.attrVal, root.nodeData, left, right, root.level, root.output, root.label) 
    return node
    # TEST OK can copy a whole decision tree
        
# pruning the tree given pruning factor
def goodpruning(root, factor, valiheader, validata):
    # accuracy before pruning
    acc0 = calaccuracy(root, valiheader, validata)
    acc1 = 0
    newroot = None
    while(acc1 < acc0):
        #Pruning
        forkroot = copyTree(root)
        newroot = pruning(forkroot, factor)
        acc1 = calaccuracy(newroot, valiheader, validata)
    return newroot
    
    
def pruning(root, factor):
    totalnodes, totalleaf = calnumber(root)
    num_prune = int(totalnodes * factor)
    attrlist = random.sample(range(totalnodes), num_prune)  
    for index in attrlist:
        pruningByindex(root, index)
    return root
    
def pruningByindex(root, index):
    if (root == None):
        return
    if (root.label == index + 1):
        root.left = None
        root.right = None
        root.output = getMajority(root)
        return   
    pruningByindex(root.left, index)
    pruningByindex(root.right, index)
    return

# get the majority class of the dataset(node's dataset)
def getMajority(node):
    counts = calCounts(node.nodeData)
    setclass = counts[len(counts) - 1]
    if (setclass[0] > setclass[3]):
            output = '1'
    else: output = '0'
    return output
    
# calculate accuracy of the tree model for a given dataset
def calaccuracy(root, header, data):
    if (data == None):
        return 0
    good = 0
    total = len(data)
    dict = mapping(header)
    for i in range(total):
        record = data[i]
        actual = record[-1]
        predicted = predictClass(root, record, dict)
        if (actual == predicted):
            good += 1
    return (float(good)/total) * 100
    
# print tree
def printTree(node):
    if (node == None):
        return
    s = node.printFormat()
    print(s)
    printTree(node.left)
    printTree(node.right)
    
# Main function
# get user's file path
trainset = input('Please type your training set path: ')
validationset = input('Please type your validation set path: ')
testset = input('Please type your test set path: ')
factor = eval(input('Please type your pruning factor: '))
# for convenience, annotate line 325-328, not annotate the following 4 lines, get the path directly
# trainset = 'training_set.csv'
# validationset = 'validation_set.csv'
# testset = 'test_set.csv'
# factor = 0.2

# Load dataset
header, data = readData(trainset)

# build the model(decision tree)
root = buildTree(data, header, 0, None, None)

# print decision tree
print('----------------------Printing decision tree constructed using ID3 algorithm----------------------------')
printTree(root)
print()

print('----------------Printing summary and results Before prunning-------------')
totalnodest, totalleaft = calnumber(root)
acc = calaccuracy(root, header, data)
print('Number of training instance = ', len(data))
print('Number of training attributes = ',len(data[0]) - 1)
print('Total number of nodes in the tree = ',totalnodest)
print('Number of leaf nodes in the tree = ',totalleaft)
print('Accuracy of the model on the training dataset = ', acc,'%')
print('Average depth of decision tree made by id3 = ', avgdepth(root))
print()
    
vheader, vdata = readData(validationset)
accv = calaccuracy(root, vheader, vdata)
print('Number of validation instance = ', len(vdata))
print('Number of validation attributes = ', len(vdata[0]) - 1)
print('Accuracy of the model on the validation dataset before pruning', accv, '%')
print()
    
theader, tdata = readData(testset)
acct = calaccuracy(root, theader, tdata)
print('Number of testing instance = ', len(tdata))
print('Number of testing attributes = ', len(tdata[0]) - 1)
print('Accuracy of the model on the testing dataset before pruning', acct, '%')
print()
    
# pruning the decision tree
labelTree(root)
newlyroot = goodpruning(root, factor, vheader, vdata)
    
# print summary and results after pruning
print('--------------After pruning, pruned based on improved accuracy on the validation dataset--------')
print('--------------Printing decision tree model After pruning----------')
printTree(newlyroot)
print()

print('--------------Printing summary and results After prunning---------')

totalnodest1, totalleaft1 = calnumber(newlyroot)
acc1 = calaccuracy(newlyroot, header, data)
print('Number of training instance = ', len(data))
print('Number of training attributes = ',len(data[0]) - 1)
print('Total number of nodes in the tree = ',totalnodest1)
print('Number of leaf nodes in the tree = ',totalleaft1)
print('Accuracy of the model on the training dataset', acc1, '%')
print()
    
accv1 = calaccuracy(newlyroot, vheader, vdata)
print('Number of validation instance = ', len(vdata))
print('Number of validation attributes = ', len(vdata[0]) - 1)
print('Accuracy of the model on the validation dataset after pruning', accv1, '%')
print()
    
acct1 = calaccuracy(newlyroot, theader, tdata)
print('Number of testing instance = ', len(tdata))
print('Number of testing attributes = ', len(tdata[0]) - 1)
print('Accuracy of the model on the testing dataset after pruning', acct1, '%')
print() 

print('---------------Build decision tree by randomly picking attribute to split------')
print('---------------Printing decision tree constructed by using random attributes----------------')
randomlyroot = randomBuildTree(data, header, 0, None, None, [])
printTree(randomlyroot)
print()

print('--------------Printing summary and results of the randomly making decision Tree---------')
rtotalnodes, rtotalleaft = calnumber(randomlyroot)
accrtr = calaccuracy(randomlyroot, header, data)
accrte = calaccuracy(randomlyroot, theader, tdata)
print('Number of training instance = ', len(data))
print('Number of training attributes = ',len(data[0]) - 1)
print('Total number of nodes in the randomly making tree = ',rtotalnodes)
print('Number of leaf nodes in the randomly making tree = ',rtotalleaft)
print('Accuracy of the model on the training dataset', accrtr, '%')
print('Accuracy of the model on the test dataset', accrte, '%')
print('Average depth of decision tree made by randomly picking attributes = ', avgdepth(randomlyroot))
            
            
            
            
            
            
    
    
    
    