import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_mode):
        super(CNN, self).__init__()
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
            self.conv1 = nn.Conv2d(20, 50, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 50, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 50, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+information' or input_mode=='protvec+gapweights'):
            self.conv1 = nn.Conv2d(101, 50, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(50,10,kernel_size_layer2, padding=pad_l2)
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

class SolvAccNet(nn.Module):
    def __init__(self, input_mode):
        super(SolvAccNet, self).__init__()
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
            self.conv1 = nn.Conv2d(20, 50, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 50, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(50,10,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(10,1, kernel_size_layer3, padding=pad_l3)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out=self.dropout((torch.selu(self.conv1(x))))
        out =self.dropout((torch.selu(self.conv2(out))))
        out = torch.selu(self.conv3(out))
        return torch.squeeze(out, 2)

class LSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = 0.2
        if(input_mode=='1hot'):
            self.embedding_dim = 20
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.embedding_dim = 100
        elif (input_mode == 'protvec+scoringmatrix'):
            self.embedding_dim = 120
        else:
            raise ValueError('wrong input mode!')

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout) #TODO batch second supposed to run faster

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)

        lstm_out, self.hidden=self.lstm(x, self.hidden)
        lstm_out=self.hidden2tag(lstm_out)
        out = F.log_softmax(lstm_out, dim=2)
        out = torch.transpose(out, 1, 2)
        return out

class biLSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(biLSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        if(input_mode=='1hot'):
            self.embedding_dim = 20
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.embedding_dim = 100
        elif (input_mode == 'protvec+scoringmatrix'):
            self.embedding_dim = 120
        else:
            raise ValueError('wrong input mode!')

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=self.num_layers, batch_first=True, bidirectional=True) #TODO batch second supposed to run faster

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device),torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device))

    def forward(self, x):
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)

        lstm_out, self.hidden=self.lstm(x, self.hidden)
        #print(lstm_out.size())
        lstm_out=self.hidden2tag(lstm_out)
        out = F.log_softmax(lstm_out, dim=2)
        out = torch.transpose(out, 1, 2)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.3)

        self.embedding_dim = 200

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False) #TODO batch second supposed to run faster

        kernel_size_layer1 = (1, 1)
        kernel_size_layer2 = (1, 7)
        kernel_size_layer3 = (1, 15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)
        pad_l3 = self._get_padding('SAME', kernel_size_layer3)

        if (input_mode == '1hot'):
            self.conv1 = nn.Conv2d(20, 150, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec' or input_mode == 'protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 150, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 150, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(150, self.embedding_dim, kernel_size_layer2, padding=pad_l2)

    def _get_padding(self, padding_type, kernel_size):
        if padding_type == 'SAME':  # zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        out = self.dropout(torch.relu(self.conv1(x)))
        out = self.dropout(torch.relu(self.conv2(out)))
        out = torch.squeeze(out, 2)
        out = torch.transpose(out, 1, 2).to(self.device)

        lstm_out, self.hidden=self.lstm(out, self.hidden)
        lstm_out=self.hidden2tag(lstm_out)
        out = F.log_softmax(lstm_out, dim=2)
        out = torch.transpose(out, 1, 2)
        return out

