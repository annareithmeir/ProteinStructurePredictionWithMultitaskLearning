import torch
import torch.nn as nn
import torch.nn.functional as F


#
# The CNNs I am using are collected here
#

class ConvNet_3_7_15(nn.Module):
    def __init__(self, input_mode):
        super(ConvNet_3_7_15, self).__init__()
        #avg dist between beta sheets influences kernel size because kernel should capture both betasheets in one frame
        kernel_size_layer1=(1,3)
        kernel_size_layer2=(1,7)
        kernel_size_layer3=(1,15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)
        pad_l3 = self._get_padding('SAME', kernel_size_layer3)
        print('pad_l1:',pad_l1)
        print('pad_l2:', pad_l2)
        print('pad_l3:', pad_l3)

        self.dropout = nn.Dropout(p=0.3)
        if(input_mode=='1hot'):
            self.conv1 = nn.Conv2d(20, 25, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec'):
            self.conv1 = nn.Conv2d(100, 25, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(25,7,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(7,4, kernel_size_layer3, padding=pad_l3)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        #out=self.dropout(F.relu(self.conv1(x)))
        #out = self.dropout(F.relu(self.conv2(out)))
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out=F.log_softmax(self.conv3(out), dim=1)
        return out

class ConvNet_1_7_15(nn.Module):
    def __init__(self, input_mode):
        super(ConvNet_1_7_15, self).__init__()
        #avg dist between beta sheets influences kernel size because kernel should capture both betasheets in one frame
        kernel_size_layer1=(1,1)
        kernel_size_layer2=(1,7)
        kernel_size_layer3=(1,15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)
        pad_l3 = self._get_padding('SAME', kernel_size_layer3)
        print('pad_l1:',pad_l1)
        print('pad_l2:', pad_l2)
        print('pad_l3:', pad_l3)

        self.dropout = nn.Dropout(p=0.3)
        if(input_mode=='1hot'):
            self.conv1 = nn.Conv2d(20, 150, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec'):
            self.conv1 = nn.Conv2d(100, 150, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(150,10,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(10,3, kernel_size_layer3, padding=pad_l3)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out=self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        out=F.log_softmax(self.conv3(out), dim=1)
        #out = F.softmax(self.conv3(out), dim=1)
        return out








