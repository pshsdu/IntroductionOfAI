# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:10:58 2018

@author: IVCL
"""

import numpy as np
import mni
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


train = list(mni.read(dataset="training", path="./mnist_data"))
test = list(mni.read(dataset="testing", path="./mnist_data"))
trLb = []
trImg = []
teLb = []
teImg = []

# Selecting data having label 0 or 1
for i in range(len(train)):
    if (train[i][0] == 0) or (train[i][0] == 1):
        trLb.append(train[i][0])
        trImg.append(train[i][1])

for i in range(len(test)):
    if (test[i][0] == 0) or (test[i][0] == 1):
        teLb.append(test[i][0])
        teImg.append(test[i][1])

print(trLb[100])
mni.show(trImg[100])

m_tr = len(trLb)
m_te = len(teLb)
# Scaling for easy of learning
for x in range(m_tr):
    trImg[x] = np.insert(trImg[x] / 255., 0, 1)
for x in range(m_te):
    teImg[x] = np.insert(teImg[x] / 255., 0, 1)

n = len(trImg[0])
# initialization of parameter vector
theta = np.random.random(n) * 0.001

alpha = 0.0005  # small learning rate
trainLoss = []
testLoss = []

for k in range(400):  # 100 is the number of epoch
    for i in range(12665):
        # Update Theta Using Gradient Decent
        theta = theta - alpha * (sigmoid(np.dot(trImg[i], theta.T)) - trLb[i]) * sigmoid(np.dot(trImg[i], theta.T)) * (1 - sigmoid(np.dot(trImg[i], theta.T))) * trImg[i]

    cost1 = 0   # Initialize Cost1
    for i in range(12665):
        # Update Cost1
        cost1 = cost1 - np.log(sigmoid(np.dot(trImg[i], theta.T)) ** trLb[i] * (1 - sigmoid(np.dot(trImg[i], theta.T))) ** (1 - trLb[i]))

    cost2 = 0   # Initialize cost2
    for i in range(2115):
        # Update Cost2
        cost2 = cost2 - np.log(sigmoid(np.dot(teImg[i], theta.T)) ** teLb[i] * (1 - sigmoid(np.dot(teImg[i], theta.T))) ** (1 - teLb[i]))

    if k % 1 == 0:
        # Append normalized Cost into Loss
        trainLoss.append(cost1 / m_tr)
        testLoss.append(cost2 / m_te)
        print(k, cost1, cost2)

# Plot Loss Function
plt.figure(0)
plt.plot(trainLoss, label='train loss')
plt.plot(testLoss, label='test loss')
plt.legend(loc='best')
plt.show()

correct = 0
for i in range(m_te):
    correct = correct + float((sigmoid(np.dot(theta, teImg[i])) > 0.5) == teLb[i])
print(float(correct) / m_te)
