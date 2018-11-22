import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities_MICHAEL as utilities

INPUT_MODE='1hot'  #protvec or 1hot
DSSP_MODE=3        #3 or 8

inputs = np.load('matrix_'+INPUT_MODE+'_train.npy', fix_imports=True)
targets=np.load('targets_'+str(DSSP_MODE)+'_train.npy', fix_imports=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', inputs.shape,' OF SIZE ',inputs[0].shape)
torch.manual_seed(42) # random seed to ensure weights are initialized to same vals.

def countStructures3(dict):

    c=0
    h=0
    e=0
    xy=0
    for sample in dict.keys():
        sequence = dict[sample][1]
        for res in range(len(sequence)):
            if(sequence[res]==1):
                c+=1
            elif(sequence[res]==2):
                h+=1
            elif(sequence[res]==3):
                e+=1
            elif(sequence[res]==4):
                xy+=1
            else:
                raise ValueError('Unknown structure', sequence[res])

    return c,h,e,xy

class ConvNet_1hot(nn.Module):
    def __init__(self):
        super(ConvNet_1hot, self).__init__()
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
        self.conv1=nn.Conv2d(100,50, kernel_size_layer1, padding=pad_l1)
        self.conv2 = nn.Conv2d(50,11,kernel_size_layer2, padding=pad_l2)
        self.conv3 = nn.Conv2d(11,4, kernel_size_layer3, padding=pad_l3)

    def _get_padding( self, padding_type, kernel_size ):
        if padding_type == 'SAME': #zero padding automatically to make kernel fit
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)

    def forward(self, x):
        out=self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        out=F.log_softmax(self.conv3(out), dim=1)
        return out

class Data_splitter():
    def __init__(self, inputs, targets, test_ratio=0.2): #80% train, 20% test
        self.inputs=inputs
        self.targets=targets
        self.test_ratio=test_ratio
        self.max_len = 0

    def split_data(self):
        train=dict()
        test=dict()
        cnt_test = 0
        cnt_train = 0
        np.random.seed(42) #to reproduce splits
        for i in range(len(self.inputs)):
            rnd = np.random.rand()

            if(len(targets[i])>self.max_len):
                self.max_len=len(targets[i])
            if(rnd>(1-self.test_ratio)):
                test[cnt_test]=(self.inputs[i], self.targets[i])
                cnt_test+=1
            else:
                train[cnt_train] = (self.inputs[i], self.targets[i])
                cnt_train+=1

        print('Size of training set: {}'.format(len(train)))
        print('Size of test set: {}'.format(len(test)))

        return train, test

    def get_max_length(self):
        return self.max_len

class Dataset_1hot( data.Dataset ):
    #LEN_FEATURE_VEC = 1

    def __init__(self, samples, max_len=None):
        self.max_len=max_len
        self.inputs, self.targets = zip(*[(inputs, targets)
                                          for _, (inputs, targets) in samples.items()])
        self.data_len = len(self.inputs)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        X=self.inputs[idx]
        #print('X.shape in getitem(): ', X.shape)
        Y=self.targets[idx]
        #print('Y.shape in getitem(): ', len(Y))
        mask_idx=np.where(np.array(Y)==4)
        mask=np.ones(len(X))
        mask[mask_idx] = 0
        #print('Y:', Y)
        #print('mask.shape in getitem(): ', mask.shape)

        if self.max_len: # if a maximum length (longest seq. in the set) was provided
            n_missing = self.max_len - len(Y) # get the number of residues which have to be padded to achieve same size for all proteins in the set
            Y    = np.pad( Y, pad_width=(0, n_missing), mode='constant', constant_values=0  )
            mask = np.pad(mask, pad_width=(0, n_missing), mode='constant', constant_values=0)
            X    = np.pad( X, pad_width=((0,n_missing),(0,0)), mode='constant', constant_values=0. )

        X = X.T
        #print('Xt.shape in getitem(): ', X.shape)
        X = np.expand_dims(X, axis=1)
        #print('Xt_expanded.shape in getitem(): ', X.shape)
        Y = np.expand_dims(Y, axis=1)
        #print('Y_expanded.shape in getitem(): ', Y.shape)
        mask = np.expand_dims(mask, axis=0)
        #print('mask_expanded.shape in getitem(): ', mask.shape)
        Y=Y.T
        #print('Yt_expanded.shape in getitem(): ', Y.shape)

        input_tensor=torch.from_numpy(X).type(dtype=torch.float)
        #print('inputtensor:', input_tensor.size())
        output_tensor=torch.from_numpy(Y)
        #print('outputtensor:', output_tensor.size())
        mask_tensor=torch.from_numpy(mask).type(dtype=torch.float)
        #print('masktensor', mask_tensor.size())
        return (input_tensor, output_tensor, mask_tensor)

    def __len__(self):
        return self.data_len

def trainNet(model, opt, crit, train_loader):
    model.train()
    for i, (X, Y_true, mask) in enumerate(train_loader):

        X = X.to(device)
        #print('train x size:',X.size())
        Y_true = Y_true.to(device)
        #print('ytrue train size: ',Y_true[0])
        mask=mask.to(device)

        opt.zero_grad()
        #print('X:',X)
        Y_raw=model(X)
        #print('train yraw size:',Y_raw[0])
        loss=crit(Y_raw, Y_true)
        loss*=mask
        loss = loss.sum() /mask.sum()
        #print('training avg loss=',loss)
        loss.backward()
        opt.step()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def testNet(model, data_loaders, crit, epoch, eval_summary):
    model.eval()
    with torch.no_grad():

        running_loss = ''
        for test_set_name, test_loader in data_loaders.items():  # for Train & Test

            loss_avg = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
            Y_true_all = np.zeros(0, dtype=np.int)
            Y_pred_all = np.zeros(0, dtype=np.int)
            confusion_matrix=np.zeros((5,5), dtype=np.int)

            percentage_correct=[]

            for X, Y, mask in test_loader:
                X=X.to(device)
                Y_true=Y.to(device)
                mask=mask.to(device)
                #print('test input: ', X)
                Y_computed = model(X)
                #print('y-raw.shape:', Y_computed.size())
                #print('adding to loss avg: ',((crit(Y_computed, Y_true)*mask).sum()) / mask.sum())
                loss_avg += ((crit(Y_computed, Y_true)*mask).sum()) / mask.sum()
                Y_pred = torch.argmax(Y_computed.data, dim=1)  # returns index of output predicted with highest probability

                #print('true structures: ', Y_true)
                #print('predicted structures: ', Y_pred)
                #print('mask: ', mask.size())
                c = 0
                w = 0
                for i in range(mask.size()[2]):
                    if(mask[0][0][i]==1):
                        if (Y_pred[0][0][i] == Y_true[0][0][i]):
                            c += 1
                        else:
                            w += 1

                percentage_correct.insert(0,c / float(c + w))

                Y_true = Y_true.view(-1).long().cpu().numpy()
                Y_pred = Y_pred.view(-1).long().cpu().numpy()
                #print('Y_true:', Y_true)
                #print('Y_pred:',Y_pred)
                np.add.at(confusion_matrix, (Y_true, Y_pred), 1)

                Y_true_all = np.append(Y_true_all, Y_true)
                Y_pred_all = np.append(Y_pred_all, Y_pred)

            loss_avg = (loss_avg / len(test_loader)).cpu().numpy()[0]
            running_loss += '{0}: {1:.3f} '.format(test_set_name, loss_avg)

            utilities.add_progress(eval_summary, test_set_name, loss_avg, epoch,
                           confusion_matrix[1:4,1:4], Y_true_all, Y_pred_all)

    print('running loss: ',running_loss)
    print('avg correct percentage:', np.average(percentage_correct))
    print('confusion mat:', confusion_matrix)



log_dir='log'
log_path=log_dir+ '/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_3-7-15_relu'
utilities.create_logdir( log_path )
d=Data_splitter(inputs, targets)
train, test=d.split_data()
max_len=d.get_max_length()

#check if classes occur around same times in train and test set
ctrain, htrain, etrain, xytrain=countStructures3(train)
trainsum=ctrain+htrain+etrain+xytrain
ctest, htest, etest, xytest=countStructures3(test)
testsum=ctest+htest+etest+xytest
print('training structure ratio:', ctrain/float(trainsum), htrain/float(trainsum), etrain/float(trainsum), xytrain/float(trainsum))
print('occurences train:', ctrain, htrain, etrain, xytrain, '-->',trainsum)
print('testing structure ratio:', ctest/float(testsum), htest/float(testsum), etest/float(testsum), xytest/float(testsum))
print('occurences test:',ctest,htest,etest,xytest,'-->',testsum)

train_set=Dataset_1hot(train, max_len)
test_set=Dataset_1hot(test, max_len)
model=ConvNet_1hot().to(device)

data_loaders = dict()
eval_summary = dict()
num_epochs=100
learning_rate=1e-3
data_loaders['Train'] = torch.utils.data.DataLoader( dataset=train_set,
                                            batch_size=32,
                                            shuffle=True,
                                            drop_last=False
                                            )
data_loaders['Test'] = torch.utils.data.DataLoader( dataset=test_set,
                                            batch_size=100,
                                            shuffle=False,
                                            drop_last=False
                                            )

print('Number of free parameters: {}'.format( count_parameters(model) ))
criterion=nn.NLLLoss(reduce=False) #reduce=False with mask
#criterion=nn.BCEWithLogitsLoss(reduce=False)
#criterion=nn.CrossEntropyLoss(reduce=False)
opt=torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# Train the model
for epoch in range(num_epochs):
    print('\n')
    print('Epoch '+str(epoch))
    trainNet(model, opt, criterion, data_loaders['Train'])
    testNet(model, data_loaders, criterion, epoch, eval_summary)
    utilities.save_performance(log_path, eval_summary)
    utilities.plot_learning(log_path, eval_summary)

utilities.plot_learning( log_path, eval_summary )
utilities.save_performance( log_path, eval_summary, final_flag=True )

torch.save(model.state_dict(), log_path+'/model.ckpt')



