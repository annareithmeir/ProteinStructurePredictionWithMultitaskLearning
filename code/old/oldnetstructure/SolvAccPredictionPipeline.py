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

    for i, (X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex) in enumerate(train_loader):  # iterate over batches
        X = X.to(device)
        Y_true = Y_solvAcc_true.to(device)

        assert (Y_true.min() >= 0 and Y_true.max() <= 1)
        mask_solvAcc = mask_solvAcc.to(device)

        opt.zero_grad()
        Y_pred_unmasked = model(X)
        loss = crit(Y_pred_unmasked, Y_true)
        loss *= mask_solvAcc
        loss = loss.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences
        total_loss_train += loss.item()

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

            for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in train_loader:  # iterate over batches
                X = X.to(device)
                Y_true_unmasked = Y_solvAcc_true.to(device)
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

        for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in test_loader: # iterate over batches
            X = X.to(device)
            Y_true_unmasked = Y_solvAcc_true.to(device)
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




INPUT_MODE='protvec_evolutionary'  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3        #3 or 8
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1_7_15_relu'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=50
LEARNING_RATE= 1e-3

weights=stats_dict[str('proportions'+str(DSSP_MODE))]
weights=weights/float(np.sum(weights))
weights=1/weights


weights_solvAcc=stats_dict['flex'] #<-----THIS IS WRONG DUE TO GETSTATS
WEIGHTS_SOLVACC=weights_solvAcc/float(np.sum(weights_solvAcc))
WEIGHTS_SOLVACC=1/WEIGHTS_SOLVACC
print('WEIGHTS_SOLVACC: ',WEIGHTS_SOLVACC)

weights_flex=stats_dict['solvAcc']
WEIGHTS_FLEX=weights_flex/float(np.sum(weights_flex))
WEIGHTS_FLEX=1/WEIGHTS_FLEX
print('WEIGHTS_FLEX: ',WEIGHTS_FLEX)

WEIGHTS=[weights, WEIGHTS_SOLVACC, WEIGHTS_FLEX]
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
        getData(prot, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex, INPUT_MODE, DSSP_MODE)

assert(len(inputs)==len(masks_struct)==len(targets_structure)==len(targets_solvAcc)==len(masks_solvAcc))

print(inputs['7AHL'].dtype, targets_structure['7AHL'].dtype, targets_solvAcc['7AHL'].dtype, targets_flexibility['7AHL'].dtype)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex, WEIGHTS)
print('train:', len(train_set))
print('test:', len(test_set))

data_loaders=DataLoader.createDataLoaders(train_set, test_set, 32, 100)


model=ConvNets.SolvAccNet(INPUT_MODE).to(device)
criterion_solvAcc=nn.NLLLoss(reduce=False) #solvAcc
opt_solvAcc=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


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
for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in sample_loader['Train']:  # iterate over batches
    X = X.to(device)
    plt.figure()
    plt.plot(Y_solvAcc_true.detach().numpy()[0][0], color='green')
    plt.plot(model(X).cpu().detach().numpy()[0][0], color='red')
    plt.savefig('sampletest.pdf')
    break



















