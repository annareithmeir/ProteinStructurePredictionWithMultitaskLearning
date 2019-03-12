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
    total_loss_train = 0 #accumulates the trainings loss
    Y_true_all = np.zeros(0, dtype=np.int) # collects all true targets
    Y_pred_all = np.zeros(0, dtype=np.int) # collects all predicted smaples

    for i, (X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex) in enumerate(train_loader): # iterate over batches

        X = X.to(device)
        Y_true = Y_struct_true.to(device)
        mask_struct=mask_struct.to(device)

        opt.zero_grad()
        Y_raw=model(X)
        loss=crit(Y_raw, Y_true)
        loss*=mask_struct
        loss = loss.sum() / float(mask_struct.sum()) # averages the loss over the structure sequences
        total_loss_train += loss.item()
        Y_pred_unmasked = torch.argmax(Y_raw.data,dim=1)  # returns index of output predicted with highest probability
        Y_pred = Y_pred_unmasked[mask_struct != 0] # applies mask with weights ( inverse to percentage of class)
        Y_true = Y_true[mask_struct != 0]


        Y_true = Y_true.view(-1).long().cpu().numpy()
        Y_pred = Y_pred.view(-1).long().cpu().numpy()

        Y_true_all = np.append(Y_true_all, Y_true)
        Y_pred_all = np.append(Y_pred_all, Y_pred)

        loss.backward()
        opt.step()


    avg_loss_train=total_loss_train / len(train_loader) #averages the total loss over the batches --> or should I rather avg over the number of samples?
    return avg_loss_train, Y_true_all, Y_pred_all

#
# Testing step
#

def testNet(model, dataloaders, crit, epoch, eval_summary, final_flag):
    test_loader=dataloaders['Test']
    train_loader=dataloaders['Train']
    model.eval()

    with torch.no_grad():
        total_loss_test=0 #accumulates total testing loss
        Y_true_all = np.zeros(0, dtype=np.int) # collects true targets
        Y_pred_all = np.zeros(0, dtype=np.int) # collects predicted targets
        confusion_matrix=np.zeros((8,8), dtype=np.int) # confusion matrix

        bootstrapped_conf_mat=[] # collects confusion matrices for each sample in last epoch

        for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in test_loader: # iterate over batches
            X = X.to(device)
            Y_true_unmasked = Y_struct_true.to(device)
            mask_struct = mask_struct.to(device)

            Y_computed = model(X)
            loss=crit(Y_computed, Y_true_unmasked)
            loss*=mask_struct
            loss = (loss.sum()) / float(mask_struct.sum())
            total_loss_test+=loss.item()
            Y_pred_unmasked = torch.argmax(Y_computed.data, dim=1)  # returns index of output predicted with highest probability
            Y_pred = Y_pred_unmasked[mask_struct != 0] # applies weighted mask
            Y_true = Y_true_unmasked[mask_struct != 0] # applies weighted mask

            if (final_flag): # last epoch, bootstrapping
                for i in range(Y_true_unmasked.size()[0]):
                    mask_sample=mask_struct[i]
                    Y_true_sample=Y_true_unmasked[i]
                    Y_pred_sample=Y_pred_unmasked[i]
                    Y_true_sample=Y_true_sample[mask_sample!=0] # applies weighted mask
                    Y_pred_sample=Y_pred_sample[mask_sample!=0] # applies weighted mask
                    #print(mask_sample.size(), Y_true_sample.size(), Y_pred_sample.size())
                    confusion_matrix_per_sample = np.zeros((8, 8), dtype=np.int)
                    np.add.at(confusion_matrix_per_sample, (Y_true_sample, Y_pred_sample), 1) # confusion matrix of one sample
                    bootstrapped_conf_mat.append(confusion_matrix_per_sample) # collect them in list

            Y_true = Y_true.view(-1).long().cpu().numpy()
            Y_pred = Y_pred.view(-1).long().cpu().numpy()
            np.add.at(confusion_matrix, (Y_true, Y_pred), 1) # confusion matrix for test step

            Y_true_all = np.append(Y_true_all, Y_true)
            Y_pred_all = np.append(Y_pred_all, Y_pred)

        avg_loss_test = (total_loss_test / len(test_loader)) # avg loss over batches --> again: rather over each sample?

        def evaluateTrainLoss():
            with torch.no_grad():
                total_loss_train = 0  # accumulates total testing loss
                Y_true_all = np.zeros(0, dtype=np.int)  # collects true targets
                Y_pred_all = np.zeros(0, dtype=np.int)  # collects predicted targets

                for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in train_loader:  # iterate over batches
                    X = X.to(device)
                    Y_true_unmasked = Y_struct_true.to(device)
                    mask_struct = mask_struct.to(device)

                    Y_computed = model(X)
                    loss = crit(Y_computed, Y_true_unmasked)
                    loss *= mask_struct
                    loss = (loss.sum()) / float(mask_struct.sum())
                    total_loss_train += loss.item()
                    Y_pred_unmasked = torch.argmax(Y_computed.data,
                                                   dim=1)  # returns index of output predicted with highest probability
                    Y_pred = Y_pred_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_true = Y_true_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_true = Y_true.view(-1).long().cpu().numpy()
                    Y_pred = Y_pred.view(-1).long().cpu().numpy()

                    Y_true_all = np.append(Y_true_all, Y_true)
                    Y_pred_all = np.append(Y_pred_all, Y_pred)

                avg_loss_train = (total_loss_train / len(
                    train_loader))  # avg loss over batches --> again: rather over each sample?

            return avg_loss_train, Y_true_all, Y_pred_all

        avg_loss_train, Y_true_all_train, Y_pred_all_train=evaluateTrainLoss()

        if(final_flag): # plotting of bootstrapping in last epoch
            print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
            utilities.plot_bootstrapping(LOG_PATH, bootstrapped_conf_mat)
        else:
            utilities_MICHAEL.add_progress(eval_summary, 'Test', avg_loss_test, epoch,
                       confusion_matrix, Y_true_all, Y_pred_all)

    return avg_loss_train, Y_true_all_train, Y_pred_all_train, avg_loss_test, Y_true_all, Y_pred_all, confusion_matrix

def getData(protein, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex, input_type, class_type):
    if (os.path.isfile(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap')):
        seq = np.load(INPUT_PATH + '/' + protein + '/' + input_type + '.npy')
        # print('unnormed')
        # print(seq)
        # normed_seq = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
        # print('normed')
        # print(normed_seq)
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
NUM_EPOCHS=100
LEARNING_RATE= 1e-3

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


model=ConvNets.ConvNet_1_7_15(INPUT_MODE).to(device)
criterion_struct=nn.NLLLoss(reduce=False) #secondary structure
opt_struct=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

#model=ConvNets.LinNet().to(device)
#criterion_solvAcc=nn.MSELoss(reduce=False) #solvAcc
#opt_solvAcc=torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)


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
    train_loss_epoch, train_true_all, train_pred_all, test_loss_epoch, test_true_all,test_pred_all, confusion_matrix_epoch=testNet(model, data_loaders, criterion_struct, epoch, eval_summary, final_flag)

    print('TRAIN LOSS:', train_loss_epoch)
    print('TEST LOSS:',test_loss_epoch)
    train_loss.append(train_loss_epoch)
    test_loss.append(test_loss_epoch)

    print(train_pred_all, train_true_all)
    print('acc score:',accuracy_score( train_true_all, train_pred_all ),accuracy_score( test_true_all, test_pred_all ))
    train_acc.append(accuracy_score( train_true_all, train_pred_all ))
    test_acc.append(accuracy_score( test_true_all, test_pred_all ))
    utilities_MICHAEL.save_performance(LOG_PATH, eval_summary)
    utilities_MICHAEL.plot_learning(LOG_PATH, eval_summary)


utilities_MICHAEL.plot_learning( LOG_PATH, eval_summary)
utilities_MICHAEL.save_performance( LOG_PATH, eval_summary, final_flag=True )

utilities.plot_loss(LOG_PATH, train_loss, test_loss, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,train_acc, test_acc, NUM_EPOCHS)
utilities.plot_confmat(LOG_PATH, confusion_matrix_epoch )
utilities.get_other_scores(confusion_matrix_epoch)
print('PARAMETERS IN MODEL:', utilities.count_parameters(model))

torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')













