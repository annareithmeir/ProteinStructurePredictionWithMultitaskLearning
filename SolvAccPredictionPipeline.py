import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities
import utilities_MICHAEL
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    plt.switch_backend('agg')  # GPU is only available via SSH (no display)

import ConvNets
import DataLoader

#
# Training step
#
def trainNet(model, opt, crit, train_loader):
    model.train()
    total_loss_train = 0  # accumulates the trainings loss
    Y_true_all = np.zeros(0, dtype=np.int)  # collects all true targets
    Y_pred_all = np.zeros(0, dtype=np.int)  # collects all predicted smaples

    for i, (X, Y_true, mask_struct, mask_solvAcc) in enumerate(train_loader):  # iterate over batches
        X = X.to(device)
        Y_true = Y_true.to(device)
        assert (Y_true.min() >= 0 and Y_true.max() <= 1)
        mask_struct = mask_struct.to(device)
        mask_solvAcc = mask_solvAcc.to(device)

        opt.zero_grad()
        Y_pred_unmasked = model(X)
        loss = crit(Y_pred_unmasked, Y_true)
        loss *= mask_solvAcc
        loss = loss.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences
        total_loss_train += loss.item()
        Y_pred = Y_pred_unmasked[mask_solvAcc != 0]  # applies mask with weights ( inverse to percentage of class)
        Y_true = Y_true[mask_solvAcc != 0]

        Y_true = Y_true.view(-1).long().cpu().numpy()
        Y_pred = Y_pred.view(-1).long().cpu().numpy()

        Y_true_all = np.append(Y_true_all, Y_true)
        Y_pred_all = np.append(Y_pred_all, Y_pred)

        loss.backward()
        opt.step()

    #print('total loss train: ', total_loss_train)


#
# Testing step
#

def testNet(model, dataloaders, crit, epoch, eval_summary, final_flag):
    def evaluateTrainLoss():
        with torch.no_grad():
            total_loss_train = 0  # accumulates total testing loss
            train_losses = []

            for X, Y, mask_struct, mask_solvAcc in train_loader:  # iterate over batches
                X = X.to(device)
                Y_true_unmasked = Y.to(device)
                mask_struct = mask_struct.to(device)
                mask_solvAcc=mask_solvAcc.to(device)

                Y_pred_unmasked = model(X)
                loss = crit(Y_pred_unmasked, Y_true_unmasked)
                loss *= mask_solvAcc
                loss = (loss.sum()) / float(mask_solvAcc.sum())
                train_losses.append(loss)
                total_loss_train += loss.item()
        avg_loss_train = np.average(train_losses)  # avg loss over batches --> again: rather over each sample?
        return avg_loss_train

    test_loader=dataloaders['Test']
    train_loader=dataloaders['Train']
    model.eval()

    with torch.no_grad():
        total_loss_test=0 #accumulates total testing loss
        test_losses=[]

        for X,  Y, mask_struct, mask_solvAcc in test_loader: # iterate over batches
            X = X.to(device)
            Y_true_unmasked = Y.to(device)
            mask_struct = mask_struct.to(device)
            mask_solvAcc=mask_solvAcc.to(device)
            Y_pred_unmasked = model(X)
            loss=crit(Y_pred_unmasked, Y_true_unmasked)
            loss*=mask_solvAcc
            loss = (loss.sum()) / float(mask_solvAcc.sum())
            test_losses.append(loss)
            total_loss_test+=loss.item()
            Y_pred = Y_pred_unmasked[mask_solvAcc != 0] # applies weighted mask
            Y_true = Y_true_unmasked[mask_solvAcc != 0] # applies weighted mask

            Y_true = Y_true.view(-1).long().cpu().numpy()
            Y_pred = Y_pred.view(-1).long().cpu().numpy()


        avg_loss_train=evaluateTrainLoss()
        print('test losses len:', len(test_losses))
        avg_loss_test=np.average(test_losses)
        print('total loss test: ', total_loss_test)


    return avg_loss_train, avg_loss_test # returns avg loss per epoch over batches

def getData(protein, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, input_type, class_type):
    if(os.path.isfile(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap')):
        flex = np.memmap(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap', dtype=np.float32, mode='r')
        tmp = {protein: flex.copy()}
        targets_flexibility.update(tmp)
        del flex

        seq=np.load(INPUT_PATH+'/'+protein+'/'+input_type+'.npy')
        #print('unnormed')
        #print(seq)
        #normed_seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
        #print('normed')
        #print(normed_seq)
        tmp={protein:seq}
        inputs.update(tmp)

        struct = np.load(INPUT_PATH + '/' + protein + '/structures_' + str(class_type) + '.npy')
        tmp = {protein: struct}
        targets_structure.update(tmp)

        solvAcc = np.memmap(TARGETS_PATH + '/dssp/' + protein.lower() + '/' + protein.lower()+'.rel_asa.memmap',  dtype=np.float32, mode='r', shape=len(struct))

        solvAcc=np.nan_to_num(solvAcc)
        assert (np.min(solvAcc)>=0 and np.max(solvAcc)<=1)
        tmp={protein: solvAcc.copy()}
        targets_solvAcc.update(tmp)
        mask_solvAcc = np.ones(len(struct))
        nans = np.argwhere(np.isnan(solvAcc.copy()))
        mask_solvAcc[nans] = 0
        tmp={protein:mask_solvAcc}
        masks_solvAcc.update(tmp)
        del solvAcc


        mask_struct = np.load(INPUT_PATH + '/' + protein + '/mask_' + str(class_type) + '.npy')
        tmp = {protein: mask_struct}
        masks_struct.update(tmp)





INPUT_MODE='protvec_evolutionary'  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3        #3 or 8
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1_7_15_relu'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=100
LEARNING_RATE= 1e-3

weights=stats_dict['proportions3']
WEIGHTS=weights/float(np.sum(weights))
WEIGHTS=1/WEIGHTS
print('WEIGHTS: ',WEIGHTS)

inputs=dict()
targets_structure=dict()
targets_solvAcc=dict()
targets_flexibility=dict()
masks_struct=dict()
masks_solvAcc=dict()


for prot in os.listdir(INPUT_PATH):
    if (len(os.listdir('preprocessing/' + prot)) == 7): #only if all input representations available
        getData(prot, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,INPUT_MODE, DSSP_MODE)

assert(len(inputs)==len(masks_struct)==len(targets_structure)==len(targets_solvAcc)==len(masks_solvAcc))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, WEIGHTS)
print('train:', len(train_set))
print('test:', len(test_set))

data_loaders=DataLoader.createDataLoaders(train_set, test_set, 32, 100)


model=ConvNets.LinNet(INPUT_MODE).to(device)
criterion_solvAcc=nn.MSELoss(reduce=False) #solvAcc
opt_solvAcc=torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)


eval_summary = dict()
final_flag=False

train_loss=[]
test_loss=[]
train_acc=[]
test_acc=[]

for epoch in range(NUM_EPOCHS):
    if(epoch==NUM_EPOCHS-1):
        final_flag=True
    print('\n')
    print('Epoch '+str(epoch))
    trainNet(model, opt_solvAcc, criterion_solvAcc, data_loaders['Train'])
    train_loss_epoch, test_loss_epoch=testNet(model, data_loaders, criterion_solvAcc, epoch, eval_summary, final_flag)

    print('TRAIN LOSS:', train_loss_epoch)
    print('TEST LOSS:',test_loss_epoch)
    train_loss.append(train_loss_epoch)
    test_loss.append(test_loss_epoch)

utilities.plot_solv_acc_loss(LOG_PATH, train_loss, test_loss)
torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')

sample_loader=DataLoader.createDataLoaders(train_set, test_set,1,1)
for X, Y_true, mask_struct, masks_solvAcc in sample_loader['Train']:  # iterate over batches
    X = X.to(device)
    print(Y_true.detach().numpy())
    print(model(X).cpu().detach().numpy())
    plt.figure()
    plt.plot(Y_true.detach().numpy()[0][0], color='green')
    plt.plot(model(X).cpu().detach().numpy()[0][0], color='red')
    plt.savefig('sampletest.pdf')
    break


















