from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        """
        This function returns a flattened tensor
        Parameters
        ----------
            x: torch.Tensor
                The tensor to flatten
        Returns
        -------
            y: torch.Tensor
                the flattened version of x
        """
        batch_size = x.shape[0]
        y = x.view(batch_size, -1)
        return y

class TestNet(nn.Module):
    '''
    This class represents the model used to classify "good" and "bad" regions of a echocargiogram according to the quality of acquisition.
    The network has 5 convolutional blocks (Convolution->ReLU->BatchNorm->MaxPooling) followed by a fully conneted layer and sigmoid function.
    Parameters
    ----------
    n_channels: int
        The number of channels in the image, e.g. for RGB images n_channels = 3, for grayscale images n_channels = 1
    n_class: int, default=1
        The number of classes in which the input image can be classified
    Attributes
    ----------
    n_channels: int
        The number of channels in the image, e.g. for RGB images n_channels = 3, for grayscale images n_channels = 1
    n_class: int, default=1
        The number of classes in which the input image can be classified
    conv_block1: ConvBlock
        The first convolution block of the network
    conv_block2: ConvBlock
        The second convolution block of the network
    conv_block3: ConvBlock
        The third convolution block of the network
    conv_block4: ConvBlock
        The fourth convolution block of the network
    conv_block5: ConvBlock
        The fifth convolution block of the network
    flatten: Flatten
        A layer which outputs a flattened tensor
    linear: nn.Linear
        A linear layer
    sig: nn.Sigmoid
        A sigmoid layer
    '''
    def __init__(self, n_channels, n_classes=1):
        super(TestNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        '''
        self.conv_block1 = ConvBlock(n_channels, 64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 128)
        self.conv_block5 = ConvBlock(128, 64)
        self.flatten = Flatten()
        '''
        self.linear0 = nn.Linear(1, 4096)
        self.linear =  nn.Linear(4096, self.n_classes) 
        #self.sig = nn.Sigmoid()
        
    def forward(self, x):
        '''
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)
        x4 = self.conv_block4(x3)
        x5 = self.conv_block5(x4)
        x6 = self.flatten(x5)
        '''
        x6 = self.linear0(x)
        x7 = self.linear(x6)
        #x8 = self.sig(x7)
        #x9= x8.squeeze(1)
        return x7
    
class ConvBlock(nn.Module):
    '''
    This class represents the model used to classify "good" and "bad" regions of a echocargiogram according to the quality of acquisition.
    The network has 5 convolutional blocks (Convolution->ReLU->BatchNorm->MaxPooling) followed by a fully conneted layer and sigmoid function.
    Parameters
    ----------
    in_channels: int
        The number of input channels of the block
    out_channels: int, default=1
        The number of output channels of the block
    Attributes
    ----------
    double_conv: nn.Sequential
        A block consisting of a convolutional layer, followed by a ReLU, folllowed by Batch Normalization, followed by a final Max Pooling layer
    '''
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self, x):
        return self.double_conv(x)