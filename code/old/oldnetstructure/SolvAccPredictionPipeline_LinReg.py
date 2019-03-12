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
from sklearn.model_selection import KFold

import ConvNets
import DataLoader

#
# Training step
#
def trainNet(model, opt, crit, train_loader):

    model.train()

    for i, (X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex) in enumerate(train_loader): # iterate over batches

        X = X.to(device)
        Y_true = Y_solvAcc_true.to(device)

        mask_solvAcc=mask_solvAcc.to(device)

        opt.zero_grad()
        Y_raw=model(X)
        loss=crit(Y_raw, Y_true)
        loss*=mask_solvAcc
        loss = loss.sum() / float(mask_solvAcc.sum()) # averages the loss over the structure sequences

        loss.backward()
        opt.step()

#
# Testing step
#

def testNet(model, dataloaders, crit, epoch, eval_summary, final_flag):
    test_loader=dataloaders['Test']
    train_loader=dataloaders['Train']
    model.eval()

    with torch.no_grad():
        total_loss_test=0 #accumulates total testing loss

        for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in test_loader: # iterate over batches
            X = X.to(device)
            Y_true_unmasked = Y_solvAcc_true.to(device)
            mask_solvAcc = mask_solvAcc.to(device)

            Y_computed = model(X)
            loss=crit(Y_computed, Y_true_unmasked)
            loss*=mask_solvAcc
            loss = (loss.sum()) / float(mask_solvAcc.sum())
            total_loss_test+=loss.item()

        avg_loss_test = (total_loss_test / len(test_loader)) # avg loss over batches --> again: rather over each sample?

        def evaluateTrainLoss():
            with torch.no_grad():
                total_loss_train = 0  # accumulates total testing loss

                for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in train_loader:  # iterate over batches
                    X = X.to(device)
                    Y_true_unmasked = Y_solvAcc_true.to(device)
                    mask_solvAcc = mask_solvAcc.to(device)

                    Y_computed = model(X)
                    loss = crit(Y_computed, Y_true_unmasked)
                    loss *= mask_solvAcc
                    loss = (loss.sum()) / float(mask_solvAcc.sum())
                    total_loss_train += loss.item()

                avg_loss_train = (total_loss_train / len(train_loader))  # avg loss over batches --> again: rather over each sample?
            return avg_loss_train

        avg_loss_train=evaluateTrainLoss()

        if (final_flag):
            predarray=np.array(Y_computed[0][0][:100].cpu().numpy())
            truearray = np.array(Y_true_unmasked[0][0][:100].cpu().numpy())
            maskarray=np.array(mask_solvAcc[0][0][:100])
            predarray=predarray[maskarray!=0]
            truearray = truearray[maskarray != 0]
            print(truearray.shape,predarray.shape,'--->')
            import matplotlib.pyplot as plt
            plt.clf()
            plt.plot(np.arange(87), predarray, color='red')
            plt.plot(np.arange(87), truearray, color='blue')
            axes = plt.gca()
            axes.set_ylim([-1, 2])
            plt.savefig('testSolvAccSampleLinReg.pdf')

    return avg_loss_train, avg_loss_test

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
        #print('unnormed')
        #print(flex)
        if (np.max(flex) - np.min(flex)!=0): #TODO: Check if this is correct
            flex = (flex - np.min(flex)) / (np.max(flex) - np.min(flex)) # normalize to [0,1]
        #print('normed')
        #print(normed_flex)
        assert (np.min(flex)>=0 and np.max(flex)<=1)
        tmp = {protein: flex.copy()}
        targets_flexibility.update(tmp)
        del flex

        solvAcc = np.memmap(TARGETS_PATH + '/dssp/' + protein.lower() + '/' + protein.lower() + '.rel_asa.memmap',
                            dtype=np.float32, mode='r', shape=len(seq))
        mask_solvAcc = np.ones(len(struct))
        nans = np.argwhere(np.isnan(solvAcc.copy()))
        mask_solvAcc[nans] = 0
        tmp = {protein: mask_solvAcc}
        masks_solvAcc.update(tmp)
        solvAcc = np.nan_to_num(solvAcc)
        assert (np.min(solvAcc) >= 0 and np.max(solvAcc) <= 1)
        tmp = {protein: solvAcc.copy()}
        targets_solvAcc.update(tmp)
        del solvAcc

        mask_struct = np.load(INPUT_PATH + '/' + protein + '/mask_' + str(class_type) + '.npy')
        tmp = {protein: mask_struct}
        masks_struct.update(tmp)


INPUT_MODE='protvec_evolutionary'  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3     #3 or 8
N_SPLITS_KFOLD=3
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1_7_15_relu'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=70
LEARNING_RATE= 0.1

weights_struct=stats_dict['proportions'+str(DSSP_MODE)]
weights_struct=weights_struct/float(np.sum(weights_struct))
weights_struct=1/weights_struct
weights_SolvAcc=[1,1]
weights_flex=[1,1,1]
WEIGHTS=[weights_struct, weights_SolvAcc, weights_flex]
print('WEIGHTS:', WEIGHTS)

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

assert(len(inputs)==len(masks_struct)==len(targets_structure)==len(targets_solvAcc))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex,WEIGHTS)
print('train:', len(train_set))
print('test:', len(test_set))

data_loaders=DataLoader.createDataLoaders(train_set, test_set, 32, 100)


model=ConvNets.SolvAccNet(INPUT_MODE).to(device)
criterion_struct=nn.MSELoss(reduce=False) #secondary structure
opt_struct=torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

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
    trainNet(model, opt_struct, criterion_struct, data_loaders['Train'])
    train_loss_epoch, test_loss_epoch=testNet(model, data_loaders, criterion_struct, epoch, eval_summary, final_flag)

    print('TRAIN LOSS:', round(train_loss_epoch,3))
    print('TEST LOSS:',round(test_loss_epoch,3))



#utilities.plot_loss(LOG_PATH, train_loss, test_loss, NUM_EPOCHS)
print('PARAMETERS IN MODEL:', utilities.count_parameters(model))

torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')













