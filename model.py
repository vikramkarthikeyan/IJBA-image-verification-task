import torch.nn as nn

# Let's start with 4 Convolution layers and work up as we need more layers
#
# Issues: The greater the number of layers, the slower is the training process and the percent improvement in 
# training accuracy decreases

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        print("Initializing model")
        print("Number of output classes:", num_classes)

        # Convolution Layer 1
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,            
                out_channels=32,           
                kernel_size=3,             
                stride=1                  
            ),                             
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )


        # Convolution Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # Convolution Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )


        # Convolution Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        # Convolution Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=96, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96)
        )


        # Convolution Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, 
                out_channels=192, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Layer 7
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, 
                out_channels=128, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )


        # Convolution Layer 8
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Layer 9
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, 
                out_channels=160, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(160)
        )


        # Convolution Layer 10
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=160, 
                out_channels=320, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(320),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(256, 1)
        ) 
    
    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        y = self.conv9(y)
        y = self.conv10(y)

        return y


class DNN(nn.Module):
    def __init__(self, num_classes):
        super(DNN, self).__init__()

        # Hidden layer for DNN
        self.out = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        
        y = self.out(x)

        return y

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

        self.features = CNN(num_classes)
        self.output = DNN(num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        return self.output(self.features(x).view(batch_size, -1))

class Base_CNN(nn.Module):

    def __init__(self, num_classes):

        super(Base_CNN, self).__init__()

        print("Initializing model")
        print("Number of output classes:", num_classes)

        # Convolution Layer 1
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,            
                out_channels=32,           
                kernel_size=3,             
                stride=1                  
            ),                             
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )


        # Convolution Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # Convolution Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )


        # Convolution Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        # Convolution Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=96, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96)
        )


        # Convolution Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, 
                out_channels=192, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Layer 7
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, 
                out_channels=128, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )


        # Convolution Layer 8
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Convolution Layer 9
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, 
                out_channels=160, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(160)
        )


        # Convolution Layer 10
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=160, 
                out_channels=320, 
                kernel_size=3,
                stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(320),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )

        self.dropout = nn.Dropout2d(p=0.4)
        
        # Hidden layer for DNN
        self.out = nn.Sequential(
            nn.Linear(320, num_classes),
            nn.Softmax(1)
        )



    def forward(self, x):

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        y = self.conv9(y)
        y = self.conv10(y)

        # Flatten output of convolution layers
        y = y.view(y.size(0), -1) 
        
        y = self.out(y)

        return y
