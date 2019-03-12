import torch
import torch.nn as nn
import torch.nn.functional as F

#
# This file handles all different network architectures.
# All networks can handle DSSP8 classification and all three variations of multi-task learning
# '1hot' : (Lx20), 'protvec' : (Lx100) --> of one seq, 'protvecevolutionary' : (Lx100) --> avg over MSA, 'protvec+scoringmatrix' : (Lx120), 'pssm' : (Lx20)
# 'multi4' : MTL of all four targets. Here, the dssp classification type states which of the two dssp variants should be
# Learned as the main task (In the thesis only dssp3 is considered as the main task). In the case of multi4-learning,
# the networks return the two structure predictions in a list. In all other cases only one structure prediction is returned.
#

class CNN(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode):
        super(CNN, self).__init__()
        kernel_size_layer1=(1,1)
        kernel_size_layer2=(1,7)
        kernel_size_layer3=(1,15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)
        pad_l3 = self._get_padding('SAME', kernel_size_layer3)

        self.dropout = nn.Dropout(p=0.3)
        self.dssp_mode=dssp_mode
        self.multi_mode=multi_mode

        if(input_mode=='1hot'):
            self.conv1 = nn.Conv2d(20, 75, kernel_size_layer1, padding=pad_l1)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 75, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 75, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'pssm'):
            self.conv1 = nn.Conv2d(20, 75, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(75, 32,kernel_size_layer2, padding=pad_l2)

        if(self.multi_mode=='multi4' or self.dssp_mode==3):
            self.conv3_3 = nn.Conv2d(32, 3, kernel_size_layer3, padding=pad_l3) # DSSP3 classification
        if(self.multi_mode=='multi4' or self.dssp_mode==8):
            self.conv3_8 = nn.Conv2d(32, 8, kernel_size_layer3, padding=pad_l3)  # DSSP8 classification

        self.conv4 = nn.Conv2d(32, 1, kernel_size_layer3, padding=pad_l3) # RSA values linear regression
        self.conv5 = nn.Conv2d(32, 1, kernel_size_layer3, padding=pad_l3) # B-factors linear regression


    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding wrt kernel
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out=self.dropout(torch.relu(self.conv1(x)))
        out = self.dropout(torch.relu(self.conv2(out)))

        out_flex= F.tanhshrink(self.conv5(out))
        out_solvAcc = torch.relu(self.conv4(out))

        if(self.multi_mode=='multi4'):
            out_struct3 = F.log_softmax(self.conv3_3(out), dim=1)
            out_struct8 = F.log_softmax(self.conv3_8(out), dim=1)

            return [out_struct3, out_struct8], torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        elif(self.dssp_mode==8):
            out_struct8 = F.log_softmax(self.conv3_8(out), dim=1)
            return out_struct8, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        else:
            out_struct3 = F.log_softmax(self.conv3_3(out), dim=1)
            return out_struct3, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)

class LSTM(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode, hidden_dim, batch_size, num_layers, device):
        super(LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.multi_mode=multi_mode
        self.dssp_mode=dssp_mode
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
        else:
            raise ValueError('wrong input mode!')

        if(self.multi_mode=='multi4' or self.dssp_mode==3):
            self.hidden2structtag_3 = nn.Linear(hidden_dim, 3)
        if(self.multi_mode=='multi4' or self.dssp_mode==8):
            self.hidden2structtag_8 = nn.Linear(hidden_dim, 8)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout)


    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        x= torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)

        lstm_out, self.hidden=self.lstm(x, self.hidden)

        lstm_out_solvacc=F.relu(self.hidden2solvacctag(lstm_out))
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2)

        lstm_out_flex=F.tanhshrink(self.hidden2flextag(lstm_out))
        out_flex = torch.transpose(lstm_out_flex, 1, 2)

        if(self.multi_mode=='multi4'):
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return [torch.unsqueeze(out_struct3, 2),torch.unsqueeze(out_struct8, 2)], out_solvAcc, out_flex
        elif(self.dssp_mode==8):
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return torch.unsqueeze(out_struct8, 2), out_solvAcc, out_flex
        else:
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            return torch.unsqueeze(out_struct3, 2), out_solvAcc, out_flex

class biLSTM(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode, hidden_dim, batch_size, num_layers, device):
        super(biLSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.multi_mode=multi_mode
        self.dssp_mode=dssp_mode
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
        else:
            raise ValueError('wrong input mode!')

        if (self.multi_mode == 'multi4' or self.dssp_mode == 3):
            self.hidden2structtag_3 = nn.Linear(hidden_dim, 3)
        if (self.multi_mode == 'multi4' or self.dssp_mode == 8):
            self.hidden2structtag_8 = nn.Linear(hidden_dim, 8)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device),torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device))

    def forward(self, x):
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        lstm_out, self.hidden=self.lstm(x, self.hidden)

        lstm_out_solvacc = F.relu(self.hidden2solvacctag(lstm_out))
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2)

        lstm_out_flex = F.tanhshrink(self.hidden2flextag(lstm_out))
        out_flex = torch.transpose(lstm_out_flex, 1, 2)

        if (self.multi_mode == 'multi4'):
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return [torch.unsqueeze(out_struct3, 2), torch.unsqueeze(out_struct8, 2)], out_solvAcc, out_flex
        elif (self.dssp_mode == 8):
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return torch.unsqueeze(out_struct8, 2), out_solvAcc, out_flex
        else:
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            return torch.unsqueeze(out_struct3, 2), out_solvAcc, out_flex

class DenseNet(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode):
        super(DenseNet, self).__init__()
        kernel_size_layer1=(1,1)
        kernel_size_layer2=(1,7)
        kernel_size_layer3=(1,15)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)
        pad_l3 = self._get_padding('SAME', kernel_size_layer3)

        self.dropout = nn.Dropout(p=0.3)
        self.dssp_mode=dssp_mode
        self.multi_mode=multi_mode

        if(input_mode=='1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(52, 22, kernel_size_layer2, padding=pad_l2)
        elif(input_mode=='protvec' or input_mode=='protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(132, 22, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 32, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(152, 22, kernel_size_layer2, padding=pad_l2)
        else:
            raise ValueError('wrong input mode!')

        if (self.multi_mode == 'multi4' or self.dssp_mode == 3):
            self.conv3_3 = nn.Conv2d(22, 3, kernel_size_layer3, padding=pad_l3)  # Classification Struct
        if (self.multi_mode == 'multi4' or self.dssp_mode == 8):
            self.conv3_8 = nn.Conv2d(22, 8, kernel_size_layer3, padding=pad_l3)  # Classification Struct

        self.conv4 = nn.Conv2d(22, 1, kernel_size_layer3, padding=pad_l3) # RSA linear regression
        self.conv5 = nn.Conv2d(22, 1, kernel_size_layer3, padding=pad_l3) # B-factors linear regression

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding wrt kernel
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out1=self.dropout(F.relu(self.conv1(x)))
        out1=torch.cat((x,out1),1)
        out2 = self.dropout(F.relu(self.conv2(out1)))

        out_flex= F.tanhshrink(self.conv5(out2))
        out_solvAcc =F.relu(self.conv4(out2))

        if (self.multi_mode == 'multi4'):
            out_struct3 = F.log_softmax(self.conv3_3(out2), dim=1)
            out_struct8 = F.log_softmax(self.conv3_8(out2), dim=1)

            return [out_struct3, out_struct8], torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        elif (self.dssp_mode == 8):
            out_struct8 = F.log_softmax(self.conv3_8(out2), dim=1)
            return out_struct8, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)
        else:
            out_struct3 = F.log_softmax(self.conv3_3(out2), dim=1)
            return out_struct3, torch.squeeze(out_solvAcc, 2), torch.squeeze(out_flex, 2)

class Hybrid(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode, hidden_dim, batch_size, num_layers, device):
        super(Hybrid, self).__init__()
        self.hidden_dim=hidden_dim
        self.multi_mode=multi_mode
        self.dssp_mode=dssp_mode
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.3)

        self.embedding_dim = 50

        if(self.multi_mode=='multi4' or self.dssp_mode==3):
            self.hidden2structtag_3 = nn.Linear(hidden_dim, 3)
        if(self.multi_mode=='multi4' or self.dssp_mode==8):
            self.hidden2structtag_8 = nn.Linear(hidden_dim, 8)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        kernel_size_layer1 = (1, 1)
        kernel_size_layer2 = (1, 7)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)

        if (input_mode == '1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 42, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec' or input_mode == 'protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 42, kernel_size_layer1, padding=pad_l1)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 42, kernel_size_layer1, padding=pad_l1)
        else:
            raise ValueError('wrong input mode!')

        self.conv2 = nn.Conv2d(42, self.embedding_dim, kernel_size_layer2, padding=pad_l2)

    def _get_padding(self, padding_type, kernel_size):
        if padding_type == 'SAME':  # zero padding wrt kernel
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device),torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim//2).to(self.device))

    def forward(self, x):
        out = self.dropout(torch.relu(self.conv1(x)))
        out = self.dropout(torch.relu(self.conv2(out)))
        out = torch.squeeze(out, 2)
        out = torch.transpose(out, 1, 2).to(self.device)

        lstm_out, self.hidden = self.lstm(out, self.hidden)

        lstm_out_solvacc = F.relu(self.hidden2solvacctag(lstm_out))
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2)

        lstm_out_flex = F.tanhshrink(self.hidden2flextag(lstm_out))
        out_flex = torch.transpose(lstm_out_flex, 1, 2)

        if (self.multi_mode == 'multi4'):
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return [torch.unsqueeze(out_struct3, 2), torch.unsqueeze(out_struct8, 2)], out_solvAcc, out_flex
        elif (self.dssp_mode == 8):
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return torch.unsqueeze(out_struct8, 2), out_solvAcc, out_flex
        else:
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            return torch.unsqueeze(out_struct3, 2), out_solvAcc, out_flex

class DenseHybrid(nn.Module):
    def __init__(self, input_mode, dssp_mode, multi_mode, hidden_dim, batch_size, num_layers, device):
        super(DenseHybrid, self).__init__()
        self.hidden_dim=hidden_dim
        self.dssp_mode=dssp_mode
        self.multi_mode=multi_mode
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.device=device
        self.dropout = nn.Dropout(p=0.3)

        self.embedding_dim = 30

        if (self.multi_mode == 'multi4' or self.dssp_mode == 3):
            self.hidden2structtag_3 = nn.Linear(hidden_dim, 3)
        if (self.multi_mode == 'multi4' or self.dssp_mode == 8):
            self.hidden2structtag_8 = nn.Linear(hidden_dim, 8)
        self.hidden2solvacctag = nn.Linear(hidden_dim, 1)
        self.hidden2flextag = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden(batch_size)
        self.lstm=nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        kernel_size_layer1 = (1, 1)
        kernel_size_layer2 = (1, 7)

        pad_l1 = self._get_padding('SAME', kernel_size_layer1)
        pad_l2 = self._get_padding('SAME', kernel_size_layer2)

        if (input_mode == '1hot' or input_mode=='pssm'):
            self.conv1 = nn.Conv2d(20, 8, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(28, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec' or input_mode == 'protvecevolutionary'):
            self.conv1 = nn.Conv2d(100, 8, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(108, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        elif (input_mode == 'protvec+scoringmatrix'):
            self.conv1 = nn.Conv2d(120, 8, kernel_size_layer1, padding=pad_l1)
            self.conv2 = nn.Conv2d(128, self.embedding_dim, kernel_size_layer2, padding=pad_l2)
        else:
            raise ValueError('wrong input mode!')

    def _get_padding(self, padding_type, kernel_size):
        if padding_type == 'SAME':  # zero padding wrt kernel
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def init_hidden(self, batch_size):
        self.batch_size=batch_size
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, x):
        out1 = self.dropout(F.relu(self.conv1(x)))
        out1 = torch.cat((x, out1), 1)
        out2 = self.dropout(F.relu(self.conv2(out1)))
        out2 = torch.squeeze(out2, 2)
        out_cnn = torch.transpose(out2, 1, 2).to(self.device)

        lstm_out, self.hidden = self.lstm(out_cnn, self.hidden)
        lstm_out_solvacc = F.relu(self.hidden2solvacctag(lstm_out))
        out_solvAcc = torch.transpose(lstm_out_solvacc, 1, 2)

        lstm_out_flex = F.tanhshrink(self.hidden2flextag(lstm_out))
        out_flex = torch.transpose(lstm_out_flex, 1, 2)

        if (self.multi_mode == 'multi4'):
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return [torch.unsqueeze(out_struct3, 2), torch.unsqueeze(out_struct8, 2)], out_solvAcc, out_flex
        elif (self.dssp_mode == 8):
            lstm_out_struct8 = F.log_softmax(self.hidden2structtag_8(lstm_out), dim=2)
            out_struct8 = torch.transpose(lstm_out_struct8, 1, 2)
            return torch.unsqueeze(out_struct8, 2), out_solvAcc, out_flex
        else:
            lstm_out_struct3 = F.log_softmax(self.hidden2structtag_3(lstm_out), dim=2)
            out_struct3 = torch.transpose(lstm_out_struct3, 1, 2)
            return torch.unsqueeze(out_struct3, 2), out_solvAcc, out_flex









