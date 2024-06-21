import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DSSRA(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DSSRA, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(
            7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(
            3, 0, 0), padding_mode='replicate', bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(
            3, 0, 0), padding_mode='replicate', bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(
            ((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.Sigmoid()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(
            0, 1, 1), padding_mode='replicate', bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(
            0, 1, 1), padding_mode='replicate', bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 358  # X1 ksc 304

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(
            1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(
            1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size,
                             out_features=n_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # SRM layer
        self.conv = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(
            1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

        self.convx = nn.Conv3d(1, 1, kernel_size=(5, 3, 3), stride=(
            1, 3, 3), padding=(2, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()
    ######################################################

    def forward(self, x, bounds=None, eps=1e-5): #SRP
        # SRM layer Spectial prior pooling
        N, C, B, _, _ = x.size()
        channel_center = x.view(
            N, C, B, -1)[:, :, :, int((x.shape[3] * x.shape[4] - 1) / 2)]
        channel_center = channel_center.unsqueeze(3)
        channel_mean = x.view(N, C, B, -1).mean(dim=3, keepdim=True)
        channel_var = x.view(N, C, B, -1).var(dim=3, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std, channel_center), dim=3)

        y = self.conv(t.unsqueeze(1)).transpose(2, 3)
        y = self.sigmoid(y)
        # y = self.bn(y)
        Y = (F.interpolate(y, size=(
            x.shape[2],x.shape[3], x.shape[4]), mode='trilinear', align_corners=False))
        Y_y = Y.reshape(Y.size(0), Y.size(2), Y.size(3), Y.size(4))
        
        ####################################################
        #Spatial prior pooling SAFP
        N, C, _, H, W = x.size()
        channel_center = x.view(x.shape[0], (x.shape[1])*(x.shape[2]), x.shape[3], x.shape[4]).mean(dim=1)
        channel_center = channel_center.unsqueeze(1)
        channel_mean = x.view(N, _, H, W).mean(dim=1, keepdim=True)
        channel_var = x.view(N,_, H, W).var(dim=1, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        s = torch.cat((channel_mean, channel_std), dim=1)

        y1 = self.convx(s.unsqueeze(1))
        y1 = self.sigmoid(y1)
        Y = (F.interpolate(y1, size=(
            x.shape[2], x.shape[3], x.shape[4]), mode='trilinear', align_corners=False))
        Y_Y = Y.reshape(Y.size(0), Y.size(2), Y.size(3), Y.size(4))
      
        #################################################################
      
        # Convolution layer 1 SFE
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))

    # End of spectral attention SFR
    #############################################
        x3 = self.conv5(x)
        x3 = self.activation5(self.bn5(x3))

        # Residual layer 2
        residual = x3
        residual = self.conv8(residual)
        x3 = self.conv6(x3)
        x3 = self.activation6(self.bn6(x3))
        x3 = self.conv7(x3)
        x3 = residual + x3

        x3 = self.activation7(self.bn7(x3))
        x3 = x3.reshape(x3.size(0), x3.size(
            1)*x3.size(2), x3.size(3), x3.size(4))  # crucial

        # concat dual spatial and spectral information
        x = torch.cat((x1,x3,Y_Y,Y_y), 1) # X2,Y,y, x3
        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        x = self.fc1(x)

        return x
  
####################################################
class ANN(nn.Module):
    """
    Attentive-Adaptive Network for Hyperspectral
    Images Classification With Noisy Labels
    Leiquan Wang , Member, IEEE, Tongchuan Zhu , Neeraj Kumar , Senior Member, IEEE, Zhongwei Li ,
    Chunlei Wu , Member, IEEE, and Peiying Zhang
    IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 61, 2023
    https://ieeexplore.ieee.org/document/10064183
    """
    def __init__(self, input_channels, patch_size,n_classes):
        super(ANN, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size= (7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size= (7, 1, 1), stride= 1, padding= (3, 0, 0), padding_mode = 'replicate', bias= True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size= (7, 1, 1), stride= 1, padding= (3, 0, 0), padding_mode = 'replicate', bias= True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish
        
        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size= (((self.feature_dim - 7) // 2 + 1), 1, 1), bias = True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()

        self.inter_size = 128
        

    #################################### SPECTRAL ENDING 
        self.sa_layer = sa_layer(channel=128, groups=64)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        self.fc1 = nn.Linear(in_features=self.inter_size,
                             out_features=n_classes)

      
    def forward(self, x, bounds=None):
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest 
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x = x1.reshape(x1.size(0),x1.size(1), x1.size(3), x1.size(4))
        

        x = self.sa_layer(x)
        x = torch.unsqueeze(x,dim=2) # b,c,b,h,w
        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        x = self.fc1(x)

        return x

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

    
