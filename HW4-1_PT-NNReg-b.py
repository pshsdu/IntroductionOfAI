# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:26:31 2020
Code for HW4-1 Regression using MLP

"""

# import library
import torch
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('/home/daeun/python_dir/IntroductionOfAI/Medical-Insurance.csv', delimiter=',', dtype='float32')

# input feature size scaling for easy of learning
for i in range(7):
    data[:, i] = data[:, i] / max(data[:, i])

''' Preparing Training and Test Data'''
# np.random.shuffle(data)  #Shuffle the data (in this exercise, it is not necessary)

trainX = torch.tensor(data[0:1000, 0:6])
trainY = torch.tensor(data[0:1000, -1].reshape(-1, 1))
testX = torch.tensor(data[1000:1338, 0:6])
testY = torch.tensor(data[1000:1338, -1].reshape(-1, 1))

# number of input layer : 6, number of hidden layer : 200, number of output layer : 1
D_in, H, D_out = 6, 100, 1

# linear regression model
# define nn model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
EPOCHS = 10000

train_loss_plot = []
test_loss_plot = []
for t in range(EPOCHS):

    # prediction
    train_pred = model(trainX)
    test_pred = model(testX)

    # calculate train loss
    # loss function에 데이터를 입력하여 구함
    train_loss = loss_fn(train_pred, trainY) / 1000.
    test_loss = loss_fn(test_pred, testY) / 338.

    # 100번에 한번씩 현재값 출력 및 출력할 창에 추가
    if t % 100 == 0:
        template = 'Epoch {}, trainLoss: {}, TestLoss: {}'
        print (template.format(t + 1, train_loss, test_loss))
        train_loss_plot.append(train_loss)
        test_loss_plot.append(test_loss)

    # Zero the gradients before running the backward pass.
    # init gradients
    model.zero_grad()
    # optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    # back propagation
    train_loss.backward()

    # optimizer.step()
    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.

    # optimizer
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# prediction
pred_train = model(trainX).detach().numpy()
pred_test = model(testX).detach().numpy()

# loss function 출력
plt.figure(1)
plt.title('201702271')
plt.plot(train_loss_plot, label='train loss')
plt.plot(test_loss_plot, label='test loss')
plt.legend(loc='best')

# train prediction 출력
ind = torch.argsort(trainY.reshape(-1))
plt.figure(2)
plt.title('201702271')
plt.plot(trainY.reshape(-1, 1)[ind], '*', label='trainY')
plt.plot(pred_train.reshape(-1, 1)[ind], '.', label='train prediction')
plt.legend(loc='best')

# test prediction 출력
ind = torch.argsort(testY.reshape(-1))
plt.figure(3)
plt.title('201702271')
plt.plot(testY.reshape(-1, 1)[ind], '*', label='testY')
plt.plot(pred_test.reshape(-1, 1)[ind], '.', label='test prediction')
plt.legend(loc='best')
