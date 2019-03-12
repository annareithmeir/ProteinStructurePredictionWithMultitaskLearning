import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCNN(nn.Module):
    def __init__(self, input_mode, dssp_mode):
        super(MultiCNN, self).__init__()
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
        self.dssp_mode=dssp_mode

        if(input_mode=='1hot'):
            self.conv1 = nn.Conv2d(20, 20, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 20, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 20, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+allpssm'):
            self.conv1 = nn.Conv2d(122, 20, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'pssm'):
            self.conv1 = nn.Conv2d(20, 20, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(20, 10,kernel_size_layer2, padding=pad_l2)

        self.conv3 = nn.Conv2d(10, self.dssp_mode, kernel_size_layer3, padding=pad_l3) #Classification Struct
        self.conv4 = nn.Conv2d(10, 1, kernel_size_layer3, padding=pad_l3) #LinReg SolvAcc
        self.conv5 = nn.Conv2d(10, 1, kernel_size_layer3, padding=pad_l3) #LinReg Flex


    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x): #x:(32,120,1,968)
        out=self.dropout(torch.selu(self.conv1(x))) #out:(32,50,1,968)
        out = self.dropout(torch.selu(self.conv2(out))) #out:(32,10,1,968)

        out_struct =F.log_softmax(self.conv3(out), dim=1) #out_struct:(32,3,1,968) --> return as (32,3,968)?
        out_flex= torch.selu(self.conv5(out)) #out_flex:(32,1,1,968) --> return as (32,1,968)
        out_solvAcc = torch.selu(self.conv4(out)) #out_solvAcc:(32,1,1,968) --> return as (32,1,968)

        return out_struct, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)

class MultiLSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(MultiLSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = 0.3
        if(input_mode=='1hot' or input_mode=='pssm'):
            self.embedding_dim = 20
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.embedding_dim = 100
        elif (input_mode == 'protvec+scoringmatrix'):
            self.embedding_dim = 120
        elif (input_mode == 'protvec+allpssm'):
            self.embedding_dim = 122
        else:
            raise ValueError('wrong input mode!')

        self.hidden2structtag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout) #TODO batch second supposed to run faster


    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x): #x:(32,120,1,968)
        x= torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2) #x:(32,968,120)

        lstm_out, self.hidden=self.lstm(x, self.hidden) #lstm_out:(32,968,32)
        lstm_out_struct=F.log_softmax(self.hidden2structtag(lstm_out), dim=2) #lstm_out_struct:(32,968,3)
        out_struct = torch.transpose(lstm_out_struct, 1, 2) #out_struct:(32,3,968)

        lstm_out_solvacc=F.selu(self.hidden2solvacctag(lstm_out)) #lstm_out_solvAcc:(32,968,1)
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2) #out_solvAcc:(32,1,968)

        lstm_out_flex=F.selu(self.hidden2flextag(lstm_out)) #lstm_out_flex:(32,968,1)
        out_flex = torch.transpose(lstm_out_flex, 1, 2) #out_flex:(32,1,968)

        #return torch.unsqueeze(out_struct,2), torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        return torch.unsqueeze(out_struct,2), out_solvAcc, out_flex

class MultibiLSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(MultibiLSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        if(input_mode=='1hot' or input_mode=='pssm'):
            self.embedding_dim = 20
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.embedding_dim = 100
        elif (input_mode == 'protvec+scoringmatrix'):
            self.embedding_dim = 120
        elif (input_mode == 'protvec+allpssm'):
            self.embedding_dim = 122
        else:
            raise ValueError('wrong input mode!')

        self.hidden2structtag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=self.num_layers, batch_first=True, bidirectional=True) #TODO batch second supposed to run faster

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device),torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device))

    def forward(self, x): #sizes like in LSTM
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        lstm_out, self.hidden=self.lstm(x, self.hidden)

        lstm_out_struct = F.log_softmax(self.hidden2structtag(lstm_out), dim=2)
        out_struct = torch.transpose(lstm_out_struct, 1, 2)

        lstm_out_solvacc = F.selu(self.hidden2solvacctag(lstm_out))
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2)

        lstm_out_flex = F.selu(self.hidden2flextag(lstm_out))
        out_flex = torch.transpose(lstm_out_flex, 1, 2)

        #return torch.unsqueeze(out_struct, 2), torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        return torch.unsqueeze(out_struct, 2), out_solvAcc, out_flex

class MultiTargetDenseNet(nn.Module):
    def __init__(self, input_mode, dssp_mode):
        super(MultiTargetDenseNet, self).__init__()
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

        self.dropout = nn.Dropout(p=0.0)
        self.dssp_mode=dssp_mode

        if(input_mode=='1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(52, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(62, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(65, 50, kernel_size_layer2, padding=pad_l2)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(132, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(142, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(145, 50, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 50, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(170, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(180, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(183, 50, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+allpssm'):
            self.conv1 = nn.Conv2d(122, 50, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(172, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(182, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(185, 50, kernel_size_layer2, padding=pad_l2)
        else:
            raise ValueError('wrong input mode!')

        self.conv3 = nn.Conv2d(50, self.dssp_mode, kernel_size_layer3, padding=pad_l3)
        self.conv4 = nn.Conv2d(50, 1, kernel_size_layer3, padding=pad_l3)
        self.conv5 = nn.Conv2d(50, 1, kernel_size_layer3, padding=pad_l3)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x): #x:(32,120,1,968)
        out1=self.dropout(F.selu(self.conv1(x))) #out1:(32,50,1,968)
        out1=torch.cat((x,out1),1) #out1:(32,170,1,968)
        out2 = self.dropout(F.selu(self.conv2(out1))) #out2:(32,10,1,968)
        out2 = torch.cat((out1, out2), 1) #out2:(32,180,1,968)
        out3 = self.dropout(F.selu(self.conv21(out2))) #out3:(32,3,1,968)
        out3 = torch.cat((out2, out3), 1) #out3:(32,183,1,968)
        out4 = self.dropout(F.selu(self.conv22(out3))) #out4:(32,50,1,968)

        out_struct =F.log_softmax(self.conv3(out4), dim=1) #out_struct:(32,3,1,968)
        out_flex= F.selu(self.conv5(out4)) #out_flex:(32,1,1,968)
        out_solvAcc =F.selu(self.conv4(out4)) #out_solvAcc:(32,1,1,968)

        return out_struct, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)

class MultiCNN_LSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(MultiCNN_LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.3)

        self.embedding_dim = 200

        self.hidden2structtag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False) #TODO batch second supposed to run faster

        kernel_size_layer1 = (1, 1)
        kernel_size_layer2 = (1, 7)
        kernel_size_layer3 = (1, 15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)

        if (input_mode == '1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 150, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec' or input_mode == 'protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 150, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 150, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+allpssm'):
            self.conv1 = nn.Conv2d(122, 150, kernel_size_layer1, padding=pad_l1)
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

    def forward(self, x): #x:(32,120,1,968)
        out = self.dropout(torch.relu(self.conv1(x))) #out:(32,150,1,968)
        out = self.dropout(torch.relu(self.conv2(out))) #out:(32,200,1,968)
        out = torch.squeeze(out, 2)
        out = torch.transpose(out, 1, 2).to(self.device) #out:(32,968, 200)

        lstm_out, self.hidden = self.lstm(out, self.hidden) #lstm_out:(32,968,6)

        lstm_out_struct = F.log_softmax(self.hidden2structtag(lstm_out), dim=2) #lstm_out_struct:(32,968,3)
        out_struct = torch.transpose(lstm_out_struct, 1, 2) #out_struct:(32,3,968)

        lstm_out_solvacc = F.selu(self.hidden2solvacctag(lstm_out)) #lstm_out_solvAcc:(32,968,1)
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2) #out_solvAcc:(32,1,968)

        lstm_out_flex = F.selu(self.hidden2flextag(lstm_out)) #lstm_out_flex:(32,968,1)
        out_flex = torch.transpose(lstm_out_flex, 1, 2) #out_flex:(32,1,968)

        #return torch.unsqueeze(out_struct, 2), torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        return torch.unsqueeze(out_struct, 2), out_solvAcc, out_flex

class MultiDenseCNN_LSTM(nn.Module):
    def __init__(self, input_mode, tagset_size, hidden_dim, batch_size, num_layers, device):
        super(MultiDenseCNN_LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.tagset_size=tagset_size
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.3)

        self.embedding_dim = 200

        self.hidden2structtag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False) #TODO batch second supposed to run faster

        kernel_size_layer1 = (1, 1)
        kernel_size_layer2 = (1, 7)
        kernel_size_layer3 = (1, 15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)

        if (input_mode == '1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(52, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(62, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(65, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec' or input_mode == 'protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(132, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(142, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(145, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 50, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(170, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(180, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(183, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+allpssm'):
            self.conv1 = nn.Conv2d(122, 50, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(172, 10, kernel_size_layer2, padding=pad_l2)
            self.conv21 = nn.Conv2d(182, 3, kernel_size_layer2, padding=pad_l2)
            self.conv22 = nn.Conv2d(185, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        else:
            raise ValueError('wrong input mode!')

    def _get_padding(self, padding_type, kernel_size):
        if padding_type == 'SAME':  # zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x): #x:(32,120,1,968)
        out1 = self.dropout(F.selu(self.conv1(x)))  # out1:(32,50,1,968)
        out1 = torch.cat((x, out1), 1)  # out1:(32,170,1,968)
        out2 = self.dropout(F.selu(self.conv2(out1)))  # out2:(32,10,1,968)
        out2 = torch.cat((out1, out2), 1)  # out2:(32,180,1,968)
        out3 = self.dropout(F.selu(self.conv21(out2)))  # out3:(32,3,1,968)
        out3 = torch.cat((out2, out3), 1)  # out3:(32,183,1,968)
        out4 = self.dropout(F.selu(self.conv22(out3)))  # out4:(32,self.embedding_dim,1,968)
        out4 = torch.squeeze(out4, 2)
        out_cnn = torch.transpose(out4, 1, 2).to(self.device)  # out:(32,968, self.embedding_dim)

        lstm_out, self.hidden = self.lstm(out_cnn, self.hidden) #lstm_out:(32,968,6)

        lstm_out_struct = F.log_softmax(self.hidden2structtag(lstm_out), dim=2) #lstm_out_struct:(32,968,3)
        out_struct = torch.transpose(lstm_out_struct, 1, 2) #out_struct:(32,3,968)

        lstm_out_solvacc = F.selu(self.hidden2solvacctag(lstm_out)) #lstm_out_solvAcc:(32,968,1)
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2) #out_solvAcc:(32,1,968)

        lstm_out_flex = F.selu(self.hidden2flextag(lstm_out)) #lstm_out_flex:(32,968,1)
        out_flex = torch.transpose(lstm_out_flex, 1, 2) #out_flex:(32,1,968)

        #return torch.unsqueeze(out_struct, 2), torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        return torch.unsqueeze(out_struct, 2), out_solvAcc, out_flex









