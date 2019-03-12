import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import Networks
import DataLoader

def trainNet(model, opt, crit, train_loader):
    model.train()
    for i, (X, Y_struct_true_unmasked, Y_solvAcc_true_unmasked, Y_flex_true_unmasked, mask_struct, mask_solvAcc, mask_flex) in enumerate(train_loader): # iterate over batches
        X=X.to(device)
        #print('xxx', X.size())
        Y_struct_true_unmasked = torch.transpose(Y_struct_true_unmasked, 1, 2)
        Y_struct_true_unmasked = torch.squeeze(Y_struct_true_unmasked, 2).to(device)
        model.zero_grad()
        model.hidden = model.init_hidden(X.size()[0])
        Y_struct_pred_unmasked = model(X)
        loss = crit(Y_struct_pred_unmasked, Y_struct_true_unmasked)
        loss = torch.unsqueeze(loss, 1)
        loss *= mask_struct.to(device)
        loss = loss.sum() / float(mask_struct.sum())  # averages the loss over the structure sequences
        #print(loss)
        loss.backward()
        opt.step()


def testNet(model, dataloaders, crit, epoch, eval_summary, final_flag):
    test_loader=dataloaders['Test']
    train_loader=dataloaders['Train']
    model.eval()

    with torch.no_grad():
        total_loss_test=0
        avg_loss_test=0

        Y_struct_true_all = np.zeros(0, dtype=np.int) # collects true targets
        Y_struct_pred_all = np.zeros(0, dtype=np.int) # collects predicted targets
        confusion_matrix=np.zeros((DSSP_MODE,DSSP_MODE), dtype=np.int) # confusion matrix
        bootstrapped_conf_mat=[] # collects confusion matrices for each sample in last epoch

        for X, Y_struct_true_unmasked, Y_solvAcc_true_unmasked, Y_flex_true_unmasked, mask_struct, mask_solvAcc, mask_flex in test_loader: # iterate over batches

            X = X.to(device)
            Y_struct_true_unmasked = torch.transpose(Y_struct_true_unmasked, 1, 2)
            Y_struct_true_unmasked = torch.squeeze(Y_struct_true_unmasked, 2).to(device)
            model.zero_grad()
            model.hidden = model.init_hidden(X.size()[0])
            Y_struct_pred_unmasked = model(X)
            loss = crit(Y_struct_pred_unmasked, Y_struct_true_unmasked)
            loss = torch.unsqueeze(loss, 1)
            loss *= mask_struct.to(device)
            loss = loss.sum() / float(mask_struct.sum())  # averages the loss over the structure sequences
            total_loss_test+=loss

            Y_struct_pred_unmasked = torch.argmax(Y_struct_pred_unmasked.data, dim=1)  # returns index of output predicted with highest probability
            mask_struct=torch.squeeze(mask_struct,1)
            Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0] # applies weighted mask
            Y_struct_true = Y_struct_true_unmasked[mask_struct != 0] # applies weighted mask

            if (final_flag): # last epoch, bootstrapping only over the test samples --> len(bootstrapped_conf_mat)==len(test_set)
                print('BOOTSTRAPPING',Y_struct_true_unmasked.size())
                for i in range(Y_struct_true_unmasked.size()[0]):
                    mask_struct_sample=mask_struct[i]
                    Y_struct_true_sample=Y_struct_true_unmasked[i]
                    Y_struct_pred_sample=Y_struct_pred_unmasked[i]
                    Y_struct_true_sample=Y_struct_true_sample[mask_struct_sample!=0] # applies weighted mask
                    Y_struct_pred_sample=Y_struct_pred_sample[mask_struct_sample!=0] # applies weighted mask
                    confusion_matrix_per_sample = np.zeros((DSSP_MODE, DSSP_MODE), dtype=np.int)
                    np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1) # confusion matrix of one sample
                    bootstrapped_conf_mat.append(confusion_matrix_per_sample) # collect them in list

            Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
            Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

            np.add.at(confusion_matrix, (Y_struct_true, Y_struct_pred), 1) # confusion matrix for test step

            Y_struct_true_all = np.append(Y_struct_true_all, Y_struct_true)
            Y_struct_pred_all = np.append(Y_struct_pred_all, Y_struct_pred)

        avg_loss_test = (total_loss_test / len(test_loader))

        def evaluateTrainLoss():
            with torch.no_grad():
                total_loss_train=0
                Y_struct_true_all = np.zeros(0, dtype=np.int)  # collects true targets
                Y_struct_pred_all = np.zeros(0, dtype=np.int)  # collects predicted targets

                for X, Y_struct_true_unmasked, Y_solvAcc_unmasked, Y_flex_true_unmasked, mask_struct, mask_solvAcc, mask_flex in train_loader:  # iterate over batches

                    X = X.to(device)
                    Y_struct_true_unmasked = torch.transpose(Y_struct_true_unmasked, 1, 2)
                    Y_struct_true_unmasked = torch.squeeze(Y_struct_true_unmasked, 2).to(device)
                    model.zero_grad()
                    model.hidden = model.init_hidden(X.size()[0])
                    Y_struct_pred_unmasked = model(X)
                    loss = crit(Y_struct_pred_unmasked, Y_struct_true_unmasked)
                    loss = torch.unsqueeze(loss, 1)
                    loss *= mask_struct.to(device)
                    loss = loss.sum() / float(mask_struct.sum())  # averages the loss over the structure sequences
                    total_loss_train+=loss

                    Y_struct_pred_unmasked = torch.argmax(Y_struct_pred_unmasked.data,dim=1)  # returns index of output predicted with highest probability
                    mask_struct = torch.squeeze(mask_struct,1)
                    Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_struct_true = Y_struct_true_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
                    Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

                    Y_struct_true_all = np.append(Y_struct_true_all, Y_struct_true)
                    Y_struct_pred_all = np.append(Y_struct_pred_all, Y_struct_pred)

                avg_loss_train = (total_loss_train / len(train_loader))
            return avg_loss_train, Y_struct_true_all, Y_struct_pred_all

        avg_loss_train, Y_true_all_train, Y_pred_all_train=evaluateTrainLoss()


        if(final_flag): # plotting of bootstrapping in last epoch
            print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
            utilities.plot_bootstrapping(LOG_PATH, bootstrapped_conf_mat)

    return avg_loss_train, Y_true_all_train, Y_pred_all_train,avg_loss_test, Y_struct_true_all, Y_struct_pred_all, confusion_matrix




def getData(protein, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex, input_type, class_type):
    if (os.path.isfile(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap')):
        seq = np.load(INPUT_PATH + '/' + protein + '/' + input_type + '.npy')
        tmp = {protein: seq}
        inputs.update(tmp)
        struct = np.load(INPUT_PATH + '/' + protein + '/structures_' + str(class_type) + '.npy')
        tmp = {protein: struct}
        targets_structure.update(tmp)

        flex = np.memmap(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap', dtype=np.float32, mode='r', shape=len(seq))
        mask_flex = np.ones(len(flex))
        nans = np.argwhere(np.isnan(flex.copy()))
        mask_flex[nans] = 0
        tmp={protein: mask_flex}
        masks_flex.update(tmp)
        flex = np.nan_to_num(flex)

        a=(flex.copy()>-1).astype(int) #make classification problem
        b=(flex.copy()>1).astype(int)
        a[np.where(b)]=2
        flex = np.array(a)

        #if (np.max(flex) - np.min(flex)!=0): #TODO: Check if this is correct
        #    flex = (flex - np.min(flex)) / (np.max(flex) - np.min(flex)) # normalize to [0,1]
        #assert (np.min(flex)>=0 and np.max(flex)<=1)
        tmp = {protein: flex.copy()}
        targets_flexibility.update(tmp)
        del flex

        solvAcc = np.memmap(TARGETS_PATH + '/dssp/' + protein.lower() + '/' + protein.lower() + '.rel_asa.memmap',
                            dtype=np.float32, mode='r', shape=len(seq))
        mask_solvAcc = np.ones(len(struct), dtype=int)
        nans = np.argwhere(np.isnan(solvAcc.copy()))
        mask_solvAcc[nans] = 0
        tmp = {protein: mask_solvAcc}
        masks_solvAcc.update(tmp)
        solvAcc = np.nan_to_num(solvAcc)


        solvAcc = (solvAcc.copy() > 0.5).astype(int) # classification problem
        solvAcc=np.array(solvAcc)

        assert (np.min(solvAcc) >= 0 and np.max(solvAcc) <= 1)
        tmp = {protein: solvAcc}
        targets_solvAcc.update(tmp)
        del solvAcc

        mask_struct = np.load(INPUT_PATH + '/' + protein + '/mask_' + str(class_type) + '.npy')
        tmp = {protein: mask_struct}
        masks_struct.update(tmp)

INPUT_MODE='protvecevolutionary'  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3       #3 or 8
N_SPLITS_KFOLD=3
LOG_PATH='log/LSTM_'+INPUT_MODE+'_'+str(DSSP_MODE)
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
#ASA_PATH='/home/mheinzinger/contact_prediction_v2/targets/structured_arrays/asa.npz'

stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=20
LEARNING_RATE= 0.6

weights=stats_dict[str('proportions'+str(DSSP_MODE))]
weights=weights/float(np.sum(weights))
WEIGHTS=1/weights
WEIGHTS=[WEIGHTS, None, None]

print('WEIGHTS: ',WEIGHTS)

inputs=dict()
targets_structure=dict()
targets_solvAcc=dict()
targets_flexibility=dict()
masks_struct=dict()
masks_solvAcc=dict()
masks_flex=dict()

for prot in os.listdir(INPUT_PATH):
    if (len(os.listdir('preprocessing/' + prot)) == 7): #only if all input representations available
        getData(prot, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex,
                INPUT_MODE, DSSP_MODE)

assert(len(inputs)==len(masks_struct)==len(targets_structure)==len(targets_solvAcc)==len(targets_flexibility))
print(len(inputs))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex,WEIGHTS)
data_loaders = DataLoader.createDataLoaders(train_set,test_set, 32, 100)

#model=ConvNets.LSTM(INPUT_MODE, DSSP_MODE, 6, 32, 1,device).to(device)
model=Networks.LSTM(INPUT_MODE, DSSP_MODE, 8, 32, 2,device).to(device)

criterion_struct=nn.NLLLoss(reduce=False)
opt_struct=torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

eval_summary = dict()
final_flag=False

train_loss_total=[]
test_loss_total=[]
train_acc=[]
test_acc=[]
confusion_matrices_total_sum=[]

for epoch in range(NUM_EPOCHS):
    if(epoch==NUM_EPOCHS-1):
        final_flag=True
    print('\nEpoch ' + str(epoch))
    trainNet(model, opt_struct, criterion_struct, data_loaders['Train'])
    train_loss_epoch, train_true_all, train_pred_all, test_loss_epoch, test_true_all, test_pred_all, confusion_matrix_epoch = testNet(
        model, data_loaders, criterion_struct, epoch, eval_summary, final_flag)

    train_loss_total.append(train_loss_epoch)
    test_loss_total.append(test_loss_epoch)

    print('acc score structure pred:', round(accuracy_score(train_true_all, train_pred_all), 2),
          round(accuracy_score(test_true_all, test_pred_all), 2))

    train_acc.append(accuracy_score(train_true_all, train_pred_all))
    test_acc.append(accuracy_score(test_true_all, test_pred_all))

utilities.plot_loss(LOG_PATH, train_loss_total, test_loss_total, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,train_acc, test_acc, NUM_EPOCHS)
utilities.plot_confmat(LOG_PATH, confusion_matrix_epoch )
utilities.get_other_scores(confusion_matrix_epoch)
print('PARAMETERS IN MODEL:', utilities.count_parameters(model))
torch.save(model.state_dict(), LOG_PATH+'/lstm.ckpt')


















