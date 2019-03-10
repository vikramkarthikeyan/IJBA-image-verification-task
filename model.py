import torch.nn as nn

# Let's start with 4 Convolution layers and work up as we need more layers
#
# Issues: The greater the number of layers, the slower is the training process and the percent improvement in 
# training accuracy decreases
class Base_CNN(nn.Module):

    def __init__(self, num_classes=200):

        super(Base_CNN, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Sequential(         # input shape (3, 64, 64)
            nn.Conv2d(
                in_channels=3,              # 3 channels for R,G,B in the image
                out_channels=128,           
                kernel_size=3,              # filter size 3*3 square kernel
                stride=1,                   
                padding=2,                  
            ),                              # output shape (128, 68, 68)
            nn.ReLU(inplace=True),                      # activation
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),    # output shape (128, 34, 34)
        )

        output = ((64 - 3 + (2 * 2)) / 1) + 1
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # Convolution Layer 2
        self.conv2 = nn.Sequential(         # input shape (128, 34, 34)
            nn.Conv2d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=2),                 # output shape (128, 38, 38)
            nn.ReLU(inplace=True),                      # activation
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),                # output shape (128, 19, 19)
        )

        output = ((output - 3 + (2 * 2)) / 1) + 1
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # Convolution Layer 3
        self.conv3 = nn.Sequential(         # input shape (128, 19, 19)
            nn.Conv2d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=2),                 # output shape (128, 23, 23)
            nn.ReLU(inplace=True),                      # activation
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),                # output shape (128, 11, 11)
        )

        output = ((output - 3 + (2 * 2)) / 1) + 1
        output = ((output - 2 + (2 * 0)) / 2) + 1

        # Convolution Layer 4
        self.conv4 = nn.Sequential(         # input shape (128, 11, 11)
            nn.Conv2d(
                in_channels=128, 
                out_channels=128, 
                kernel_size=3,
                stride=1,
                padding=2),                 # output shape (128, 15, 15)
            nn.ReLU(inplace=True),                      # activation
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),                # output shape (128, 7, 7)
        )

        output = ((output - 3 + (2 * 2)) / 1) + 1
        output = ((output - 2 + (2 * 0)) / 2) + 1

        output = 128 * output * output
        
        # Hidden layer for DNN
        self.hidden = nn.Sequential(
            nn.Linear(output, 1024),
            nn.Softmax(1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5)
        )

        # Output layer with a Linear Transformation
        self.out = nn.Linear(1024, num_classes)


    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        # Flatten output of convolution layers
        y = y.view(y.size(0), -1) 

        y = self.hidden(y)
        y = self.out(y)

        return y
