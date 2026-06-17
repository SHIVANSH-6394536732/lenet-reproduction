from multiprocessing.managers import pool
import numpy as np
from scratch.layers import Conv2D, MaxPool2D, FullyConnected
from scratch.activations import ReLu


class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(num_filters= 6 , filter_size= 5)
        self.relu1 = ReLu()
        self.pool1 = MaxPool2D(pool_size= 2)

        self.conv2 = Conv2D(num_filters= 16 , filter_size= 5 , input_channels= 6)
        self.relu2 = ReLu()
        self.pool2 = MaxPool2D(pool_size= 2)

        self.fc1 = FullyConnected(input_size= 4*4*16 , output_size= 120)
        self.relu3 = ReLu()

        self.fc2 = FullyConnected(input_size= 120 , output_size= 84)
        self.relu4 = ReLu()

        self.fc3 = FullyConnected(input_size=84, output_size=10)

    def forward(self,image):

        out = self.conv1.forward(image)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)


        out = self.fc1.forward(out)
        out = self.relu3.forward(out)

        out = self.fc2.forward(out)
        out = self.relu4.forward(out)

        out = self.fc3.forward(out)

        return out






