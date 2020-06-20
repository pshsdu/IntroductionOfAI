# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 정의
mnist_train = datasets.MNIST('./mnist_data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
mnist_test = datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

# 데이터셋을 DataLoader에 적용
train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=1000, shuffle=True)

# GPU 또는 CPU 사용을 위한 device 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 모델 정의
class CNN_Model(nn.Module):

  # CNN_Model Class가 호출되었을 때 가장 먼저 실행되는 함수
  # CNN 모델을 정의해놓음
  # input으로 input channel의 갯수, output channel의 갯수, feature channel의 갯수, convolution을 실행할 횟수
    def __init__(self, input_channel_num, output_channel_num, feature_channel_num, num_conv):
        super(CNN_Model, self).__init__()

        # input channel 갯수가 input_channel_num이고 ouput channel 갯수가 feature_channel_num인 CNN을 실행한다.
        # kernal size는 3이고 padding을 함으로써 인풋 행렬의 크기를 늘린다.
        self.in_conv = nn.Sequential(
            nn.Conv2d(input_channel_num, feature_channel_num, kernel_size=3, padding=1),
            nn.ReLU()
        )

        convs = list()
        chan = feature_channel_num

        # feature_channel_num의 갯수를 chan으로 받고 input channel 갯수가 chan ouput channel 갯수가 chan*2 CNN을 실행한다.
        # 실행된 결과에 max pooling을 하여 사이즈를 줄여준다.
        for _ in range(num_conv):
            convs.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
            ))
            chan = chan * 2

        self.conv = nn.Sequential(*convs)

        self.out_conv = nn.Conv2d(chan, output_channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)

        # global average pooling
        x = torch.mean(x, dim=(-1, -2))
        x = F.softmax(x, dim=-1)
        return x

# CNN 모델 정의
class CNN_Model_MLP(nn.Module):

  # CNN_Model Class가 호출되었을 때 가장 먼저 실행되는 함수
  # CNN 모델을 정의해놓음
  # input으로 input channel의 갯수, output channel의 갯수, feature channel의 갯수, convolution을 실행할 횟수
    def __init__(self, input_channel_num, feature_channel_num, num_conv):
        super(CNN_Model_MLP, self).__init__()

        # input channel 갯수가 input_channel_num이고 ouput channel 갯수가 feature_channel_num인 CNN을 실행한다.
        # kernal size는 3이고 padding을 함으로써 인풋 행렬의 크기를 늘린다.
        self.in_conv = nn.Sequential(
            nn.Conv2d(input_channel_num, feature_channel_num, kernel_size=3, padding=1),
            nn.ReLU()
        )

        convs = list()
        chan = feature_channel_num

        # feature_channel_num의 갯수를 chan으로 받고 input channel 갯수가 chan ouput channel 갯수가 chan*2 CNN을 실행한다.
        # 실행된 결과에 max pooling을 하여 사이즈를 줄여준다.
        for _ in range(num_conv):
            convs.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
            ))
            chan = chan * 2

        self.conv = nn.Sequential(*convs)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

        # self.out_conv = nn.Conv2d(chan, output_channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(x)
        x = self.mlp(x)
        x = F.softmax(x, dim=-1)
        return x


param = [1, 10, 64, 2]
model = CNN_Model(*param).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

param_mlp = [1, 64, 2]
model_mlp = CNN_Model_MLP(*param_mlp).to(device)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:5d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()  # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n 201702271 Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 학습 및 테스트 실행
for epoch in range(10):
    print("global agerage model")
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

    print("mlp model")
    train(model_mlp, train_loader, optimizer_mlp, epoch)
    test(model_mlp, test_loader)
