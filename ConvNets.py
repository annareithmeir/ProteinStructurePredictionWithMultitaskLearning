import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.dropout = nn.Dropout(p=0.6)
        if(input_mode=='1hot'):
            self.conv1 = nn.Conv2d(20, 150, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec' or input_mode=='protvec_evolutionary'):
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
        out=self.dropout(torch.tanh(self.conv1(x)))
        out = self.dropout(torch.tanh(self.conv2(out)))
        out=F.log_softmax(self.conv3(out), dim=1)
        #out = F.softmax(self.conv3(out), dim=1)
        return out

class LinNet(nn.Module):
    def __init__(self, input_mode):
        super(LinNet, self).__init__()
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
        elif(input_mode=='protvec' or input_mode=='protvec_evolutionary'):
            self.conv1 = nn.Conv2d(100, 150, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(150,10,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(10,3, kernel_size_layer3, padding=pad_l3)
        self.fc1=nn.Linear(968*10,968*2)
        self.fc2 = nn.Linear(968 * 2, 968)

        #self.fc2 = nn.Linear(5, 1)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        #print('###')
        #print('1:','analysis:', torch.max(x), torch.min(x), x.size())
        out=self.dropout(torch.tanh(self.conv1(x)))
        #print('1.5:', 'analysis:', torch.max(x), torch.min(x), out.size())
        out =self.dropout(torch.tanh(self.conv2(out)))
        #print('2','analysis:', torch.max(out), torch.min(out), out.size())
        out = out.reshape(out.size(0), -1)
        #print('3', 'analysis:', torch.max(out), torch.min(out), out.size())
        out = self.dropout(torch.tanh(self.fc1(out)))
        out = torch.tanh(self.fc2(out))
        out=out[:, None, :]
        #print('5','analysis:', torch.max(out), torch.min(out), out.size())
        return out

class MultiTargetNet(nn.Module):
    def __init__(self, input_mode):
        super(MultiTargetNet, self).__init__()
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
        elif(input_mode=='protvec' or input_mode=='protvec_evolutionary'):
            self.conv1 = nn.Conv2d(100, 150, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(150, 10,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(10, 3, kernel_size_layer3, padding=pad_l3)
        self.conv4 = nn.Conv2d(10, 2, kernel_size_layer3, padding=pad_l3)
        self.conv5 = nn.Conv2d(10, 3, kernel_size_layer3, padding=pad_l3)

        self.fc1=nn.Linear(968*10,968*2)
        self.fc2 = nn.Linear(968 * 2, 968)

        self.fc3=nn.Linear(968*10,968*2)
        self.fc4 = nn.Linear(968 * 2, 968)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out=F.selu(self.conv1(x))
        out = F.selu(self.conv2(out))

        '''
        out_solvAcc = out.reshape(out.size(0), -1)
        out_solvAcc = torch.relu(self.fc1(out_solvAcc))
        out_solvAcc = torch.relu(self.fc2(out_solvAcc))
        out_solvAcc = out_solvAcc[:, None, :]
        '''

        out_struct =F.log_softmax(self.conv3(out), dim=1)

        out_flex= F.log_softmax(self.conv5(out), dim=1)

        out_solvAcc =F.log_softmax(self.conv4(out), dim=1)

        '''
        out_flex = out.reshape(out.size(0), -1)
        out_flex = torch.relu(self.fc3(out_flex))
        out_flex = torch.relu(self.fc2(out_flex))
        out_flex = out_flex[:, None, :]
        '''

        return out_struct, out_solvAcc, out_flex









