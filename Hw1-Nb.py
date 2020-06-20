# -*- coding: utf-8 -*-
"""HW#1 Code
The data set in this home work is from
http://https://www.kaggle.com/datasets
This dataset is about  Mecical Insurance Charges in US
Homework-0 code for getting familiar with numpy
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('Medical-Insurance.csv', delimiter=',')

'''
data is a (1338,7) nparray. 1338 is the number of person
Each row in data (X1, X2, ..., X7) is for each person.
X1 = age
X2 = sex(0 for female, 1 for male)
X3 = BMI (weight(kg)/(height(m)*height(m))
X4 = number of children 
X5 = Smoker or not (1 for smoker, 0 for n0n-smoker)
X6 = Region (southwest(1),northwest(2),northeast(3),southeast(4))
X7 = Medical Charges in dollar
'''
'''Preparing Training and Test Data'''

# input feature size scaling for easy of learning
for i in range(7):
    data[:, i] = data[:, i] / max(data[:, i])

data = np.insert(data, 0, 1, axis=1)  # insert 1 for each row, now (1338,7) array
np.random.shuffle(data)  # Shuffle the data
data = data.T  # now (8,1338) array
data.shape  # print the dimension od data
trainX = data[0:7, 0:1000]  # (7,1000), 1000 is th number of training data
trainY = data[-1, 0:1000]  # (1000,)
testX = data[0:7, 1000:1338]  # (7,338), 338 is th number of testing data
testY = data[-1, 1000:1338]  # (338,)

# Analytic Solution(parameter, min-cost)
X = trainX.T
Y = trainY
A = np.linalg.inv(np.matmul(X.T, X))
B = np.matmul(X.T, Y)
theta = np.matmul(A, B)
cost = 0.5 * np.linalg.norm(np.matmul(X, theta) - Y) ** 2.
print(cost)  # printing global min loss

theta = np.random.normal(0, 1, (7))  # parameter random initialization

epoch = 1000
alpha = 0.0005  # small learning rate
trainLoss = []
testLoss = []

for k in range(epoch):
    for i in range(1000):
        # Update Theta
        theta = theta - alpha * (np.matmul(trainX[:, i], theta) - trainY[i]) * trainX[:, i]

    cost_tr = 0  # initial cost for each epoch

    for i in range(1000):
        # Update Training Cost
        cost_tr = cost_tr + 0.5 * (np.matmul(trainX[:, i], theta) - trainY[i]) ** 2.

    cost_te = 0  # initial cost for each epoch

    for i in range(338):
        # Update Test Cost
        cost_te = cost_te + 0.5 * (np.matmul(testX[:, i], theta) - testY[i]) ** 2.

    if k % 100 == 0:
        trainLoss.append(cost_tr / 1000)
        testLoss.append(cost_te / 338)

print(cost_tr)  # printing final loss
plt.figure(0)
plt.plot(trainLoss, label='train loss')
plt.plot(testLoss, label='test loss')
plt.legend(loc='best')
plt.show()
print(np.argmin(testLoss))

pred = []
for i in range(338):
    pred.append(np.dot(theta, testX[:, i]))
pred = np.asarray(pred)  # converting list to numpy array
ind = np.argsort(testY)
plt.figure(1)
plt.plot(pred[ind], '*', label='test prediction')
plt.plot(testY[ind], '.', label='testY')
plt.legend(loc='best')
plt.show()

pred = []
for i in range(1000):
    pred.append(np.dot(theta, trainX[:, i]))
pred = np.asarray(pred)  # converting list to numpy array
ind = np.argsort(trainY)

plt.figure(2)
plt.plot(pred[ind], '*', label='train prediction')
plt.plot(trainY[ind], '.', label='trainY')
plt.legend(loc='best')
plt.show()
