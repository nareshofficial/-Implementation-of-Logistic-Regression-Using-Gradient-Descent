# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NARESH.P.S
RegisterNumber:  2122223040127
*/
```
```
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/classes/ML/New folder/Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('sl_no',axis=1)

#catogorising col for further labegling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the colums
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#display dependent variables
Y

#initialize the model parameter
theta = np.random.randn(X.shape[1])
y=Y

#define the sigmoid function 
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define the loss function 
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

#define the gradient descent algorithm
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

#train the model
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

# make the predictions
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

#evaluate the model
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy : ",accuracy)
print(y_pred)
print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:
dataset:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/b9b53ced-ff22-48f1-bd01-9d8be027192c)

dataset.dtypes:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/8e6091f9-b71f-4782-b7d1-6444b5d83616)

dataset:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/9a83b9f9-48a3-48b2-8315-c7e9d74435bf)



Y:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/760993ab-1e4b-479e-aa2f-ec5e984f5b35)

y_pred:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/b5aeb284-3645-4b8a-aaee-9436ec55bacc)

Y:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/36d4cbb7-e451-48a0-89f6-c61a678fb1c5)

y_prednew:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/20def8a9-c007-4bb9-94b1-e385f3accc37)

y_prednew:

![image](https://github.com/23004513/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138973069/7d9ac4aa-54af-43d2-89f0-51ce992b72f4)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

