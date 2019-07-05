<<<<<<< HEAD
<<<<<<< HEAD
# Big-Data-Machine-Learning
Academic project in big data and machine learning class
=======
# Decision-Tree
Using Python to implemente ID3 algorithm and pruning on training dataset and validation dataset

What I have done:
•	Developed a decision tree classifier to predict high-risk patients of a newly admitted patients by training data containing 601 instances and 20 features.
•	Implemented machine learning algorithm ID3 to build the model, achieved accuracy of 99.8% on training data, accuracy of 75.9% on testing data.
•	Optimized the classifier by applying reduced error pruning technique to reduce overfitting, enhanced prediction performance to 80% on testing data.


File structure:
1. myID3.py: this file contains all the code we need.
2. Running_result: this file contains the result when I run myid3.py, including tree model representation and summary and results for id3 tree model, pruned id3 tree model, randomly attributes constructed tree model.
3. training_set.csv: training dataset
4. validation_set.csv: validation dataset
5. test_set.csv: test dataset
6. README.txt

How to run code:
Open terminal(in MAC or putty in windows) and set the path to where the submitted code is located at

The program would ask users to 
Please type your training set path: 
Please type your validation set path: 
Please type your test set path: 
Please type your pruning factor: 

Then users need to type the following respectively
training_set.csv
validation_set.csv
test_set.csv
0.2

The pruning factor can be any number less than 1, larger than 0.
>>>>>>> old1/master
=======
# Artificial-Neural-Network-for-Car-Evaluation
This project built a ANN model to determine if a car is good or not with given parameters. 

What I have done are:
•	Constructed and trained a neural network to evaluate cars using dataset with 1728 instances, 6 features.
•	Optimized the ANN by preprocessing raw data with Pandas and Numpy, which cleaned, standardized, converted data to ensure good work with numeric and scaled data.
•	Applied backpropagation algorithm and replaced activation function sigmoid with ReLU to optimize the model, which improved accuracy to 98%.

How to run:
Use python to run ANN.py, the input is the processed data, which is car_processed.txt.
>>>>>>> old2/master
