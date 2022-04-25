import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ['UNet', 'NestedUNet', 'VolumeEstimation']


class VGGBlock(nn.Module):
    '''
        Consists of 2 Convolution Layers with ReLU and batch normalization between them. 
        params: Setting the filter size through the number of channels in input, middle and the final output.
    '''
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    '''
    Unet Architecture (Reference - https://arxiv.org/pdf/1505.04597.pdf)
    params: number of classes and input channels.
    '''	
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    '''
    Unet++ Architecture (Reference - https://arxiv.org/pdf/1807.10165.pdf)
    params: number of classes and input channels.
    '''	
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

# class RegressionBlock(nn.Module):
#     #def __init__(self, in_channels, middle_channels, out_channels):
#     def __init__(self, numSlices):
#         super().__init__()
#         #self.fc1 = nn.Linear(in_channels, middle_channels)
#         #self.fc2 = nn.Linear(middle_channels, out_channels)

#         # Assumes images are 96 x 96 pixels

#         # 1-4
#         self.conv1 = nn.Conv2d(in_channels=numSlices,out_channels=(numSlices*2),kernel_size=5,stride=1,padding=0,bias=False)
#         nn.init.kaiming_uniform_(self.conv1.weight)
#         self.normLay1 = nn.LayerNorm(normalized_shape=[(numSlices*2),88,88]) # May need to change y and z in [x,y,z]
#         self.relu1 = nn.LeakyReLU()
#         self.maxpl1 = nn.MaxPool2d(kernel_size=2,stride=2)

#         # 5-9
#         self.depth1 = nn.Conv2d(in_channels=(numSlices*2),out_channels=(numSlices*2),kernel_size=5,stride=2,padding=0,groups=(numSlices*2),bias=False)
#         nn.init.kaiming_uniform_(self.depth1.weight)
#         self.normLay2 = nn.LayerNorm(normalized_shape=[(numSlices*2),20,20]) # Will likely have to change y and z in [x,y,z]
#         self.relu2 = nn.LeakyReLU()
#         self.maxpl2 = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.point1 = nn.Conv2d(in_channels=(numSlices*2),out_channels=(numSlices*4),kernel_size=1,stride=1,padding=0,bias=True)
#         nn.init.xavier_uniform_(self.point1.weight)
#         self.point1.bias.data.fill_(0)

#         # 10
#         self.fc1 = nn.Conv2d(in_channels=(numSlices*2),out_channels=1,kernel_size=4,stride=1,padding=0,bias=True) # Will likely need to change kernel_size
#         nn.init.xavier_uniform_(self.fc1.weight)
#         self.fc1.bias.data.fill_(0)

#     def forward(self, x):
#         #mid = self.fc1(x)
#         #mid = F.relu(mid)
#         #out = self.fc2(mid)
#         #out = torch.flatten(out)

#         # 1-4
#         out = self.conv1(x)
#         out = self.normLay1(out)
#         out = self.relu1(out)
#         out = self.maxpl1(out)

#         # 5-9
#         out = self.depth1(out)
#         out = self.normLay2(out)
#         out = self.relu2(out)
#         out = self.maxpl2(out)
#         out = self.point1(out)

#         # 10
#         out = self.fc1(out)
        
#         return out

class FCBlock(nn.Module):
    def __init__(self, in_channels, middle_one_channels, middle_two_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, middle_one_channels)
        self.fc2 = nn.Linear(middle_one_channels, middle_two_channels)
        self.fc3 = nn.Linear(middle_two_channels, out_channels)

    def forward(self, x):
        mid = self.fc1(x)
        mid = F.relu(mid)
        mid = self.fc2(mid)
        mid = F.relu(mid)
        out = self.fc3(mid)
        out = torch.flatten(out)
        
        return out

class VolumeEstimation(nn.Module):
    '''
    Unet++ Architecture (Reference - https://arxiv.org/pdf/1807.10165.pdf)
    params: number of classes and input channels.
    '''	
    def __init__(self, shortSegModel, longSegModel, useLong, numSlices, poolingMethod='average', **kwargs):
        super().__init__()

        self.shortSegModel = shortSegModel
        self.longSegModel = longSegModel
        self.useLong = useLong
        self.numSlices = numSlices + 1 if self.useLong else numSlices
        self.poolingMethod = poolingMethod

        # Volume Regression Layer
        #self.regression = RegressionBlock(self.numSlices)
        self.regression = FCBlock(self.numSlices*9216, 9216, 16, 2)
        self.regression.cuda()

    def poolingFusion(self, sliceMasks):
        if self.poolingMethod == 'average':
            return torch.mean(sliceMasks, 0)
        elif self.poolingMethod == 'max':
            elements, _ = torch.max(sliceMasks, 0)
            return elements
        elif self.poolingMethod == 'min':
            elements, _ = torch.min(sliceMasks, 0)
            return elements

    def forward(self, input):
        long_imgs, short_imgs = input
        descriptors = torch.Tensor()

        if self.useLong:
            longSegs = []
            for long in long_imgs:
                longSegs.append(self.longSegModel(long.cuda()).data.cpu().numpy())
                #torch.cuda.empty_cache()
            longSegs = torch.as_tensor(np.array(longSegs))
            longSegs = torch.squeeze(longSegs)
            longDesciptor = self.poolingFusion(longSegs)
            descriptors = torch.cat([descriptors,longDesciptor])

        for slice in short_imgs:
            sliceSegs = []
            for short in slice:
                sliceSegs.append(self.shortSegModel(short.cuda()).data.cpu().numpy())
                #torch.cuda.empty_cache()
            sliceSegs = torch.as_tensor(np.array(sliceSegs))
            sliceSegs = torch.squeeze(sliceSegs)
            sliceDesciptor = self.poolingFusion(sliceSegs)
            descriptors = torch.cat([descriptors,sliceDesciptor])

        #descriptors = torch.as_tensor(np.array(longSegs))
        regInput = torch.flatten(descriptors)
        regInput = regInput.cuda()
        output = self.regression(regInput)

        return output