import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import Networks
import DataLoader
import MultiTargetPredictionPipeline
import utilities

#
# Training step
#
def trainNet(model, opts, crits, train_loader):

    model.train()
    total_struct_loss_train = 0 #accumulates the trainings loss
    total_solvAcc_loss_train = 0
    total_flex_loss_train = 0
    Y_struct_true_all = np.zeros(0, dtype=np.int) # collects all true targets
    Y_struct_pred_all = np.zeros(0, dtype=np.int) # collects all predicted smaples

    for i, (X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex) in enumerate(train_loader): # iterate over batches

        X = X.to(device)
        Y_struct_true = Y_struct_true.to(device)
        Y_solvAcc_true = Y_solvAcc_true.to(device)
        Y_flex_true = Y_flex_true.to(device)
        mask_solvAcc=mask_solvAcc.to(device)
        mask_struct=mask_struct.to(device)
        mask_flex = mask_flex.to(device)

        opts[0].zero_grad()
        #opts[1].zero_grad()
        #opts[2].zero_grad()
        Y_struct_raw, Y_solvAcc_raw, Y_flex_raw=model(X)


        loss_struct=crits[0](Y_struct_raw, Y_struct_true)
        loss_struct*=mask_struct
        loss_struct = loss_struct.sum() / float(mask_struct.sum()) # averages the loss over the structure sequences
        total_struct_loss_train += loss_struct.item()

        loss_solvAcc = crits[1](Y_solvAcc_raw, Y_solvAcc_true)
        loss_solvAcc *= mask_solvAcc
        loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences
        total_solvAcc_loss_train += loss_solvAcc.item()

        loss_flex= crits[2](Y_flex_raw, Y_flex_true)
        loss_flex*=mask_flex

        if(mask_flex.sum()!=0): #we have one sample where the mask is zero everywhere (2EF4)
            loss_flex=loss_flex.sum() / float(mask_flex.sum())
        else:
            loss_flex=torch.zeros(1)
        total_flex_loss_train += loss_flex.item()

        loss=loss_struct
        #loss=loss_struct


        Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data,dim=1)  # returns index of output predicted with highest probability
        Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0] # applies mask with weights ( inverse to percentage of class)
        Y_struct_true = Y_struct_true[mask_struct != 0]

        Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
        Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

        Y_struct_true_all = np.append(Y_struct_true_all, Y_struct_true)
        Y_struct_pred_all = np.append(Y_struct_pred_all, Y_struct_pred)

        loss.backward()
        opts[0].step()
        #opts[1].step()
        #opts[2].step()

    avg_struct_loss_train=total_struct_loss_train / len(Y_struct_pred_all) #averages the total loss over total number of residues in all batches
    avg_solvAcc_loss_train = total_solvAcc_loss_train / len(Y_struct_pred_all)
    avg_flex_loss_train = total_flex_loss_train / len(Y_struct_pred_all)

    return avg_struct_loss_train, avg_solvAcc_loss_train, avg_flex_loss_train, Y_struct_true_all, Y_struct_pred_all

#
# Testing step
#

def testNet(model, dataloaders, crits, epoch, eval_summary, final_flag):
    test_loader=dataloaders['Test']
    train_loader=dataloaders['Train']
    model.eval()

    with torch.no_grad():
        total_losses_test=[0,0,0, 0]
        avg_losses_test=[0,0,0,0] #[struct,solvAcc,flex,sum]

        Y_struct_true_all = np.zeros(0, dtype=np.int) # collects true targets
        Y_struct_pred_all = np.zeros(0, dtype=np.int) # collects predicted targets
        confusion_matrix=np.zeros((DSSP_MODE,DSSP_MODE), dtype=np.int) # confusion matrix
        bootstrapped_conf_mat=[] # collects confusion matrices for each sample in last epoch

        for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in test_loader: # iterate over batches
            X = X.to(device)
            Y_struct_true_unmasked = Y_struct_true.to(device)
            Y_solvAcc_true = Y_solvAcc_true.to(device)
            Y_flex_true = Y_flex_true.to(device)
            mask_struct = mask_struct.to(device)
            mask_solvAcc=mask_solvAcc.to(device)
            mask_flex = mask_flex.to(device)

            Y_struct_raw, Y_solvAcc_raw, Y_flex_raw = model(X)

            loss_struct=crits[0](Y_struct_raw, Y_struct_true_unmasked)
            loss_struct*=mask_struct
            loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
            total_losses_test[0]+=loss_struct.item()

            loss_solvAcc = crits[1](Y_solvAcc_raw, Y_solvAcc_true)
            loss_solvAcc *= mask_solvAcc
            loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences
            total_losses_test[1] += loss_solvAcc.item()

            loss_flex = crits[2](Y_flex_raw, Y_flex_true)
            loss_flex *= mask_flex
            loss_flex = loss_flex.sum() / float( mask_flex.sum())  # averages the loss over the structure sequences
            total_losses_test[2] += loss_flex.item()

            total_losses_test[3] += loss_struct.item()
            print('TEST: solvAcc loss-', round(loss_solvAcc.item(),3), ' struct loss-', round(loss_struct.item(),3),' flex loss-', round(loss_flex.item(),3), ' overall loss-', round(loss_struct.item(),3))

            Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data, dim=1)  # returns index of output predicted with highest probability
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

        avg_losses_test[0] = (total_losses_test[0] / len(test_loader))
        avg_losses_test[1] = (total_losses_test[1] / len(test_loader))
        avg_losses_test[2] = (total_losses_test[2] / len(test_loader))
        avg_losses_test[3] = (total_losses_test[3] / len(test_loader))  # avg loss over total residues

        def evaluateTrainLoss():
            with torch.no_grad():
                total_losses_train=[0,0,0,0]
                avg_losses_train=[0,0,0,0]
                Y_struct_true_all = np.zeros(0, dtype=np.int)  # collects true targets
                Y_struct_pred_all = np.zeros(0, dtype=np.int)  # collects predicted targets

                for X, Y_struct, Y_solvAcc, Y_flex, mask_struct, mask_solvAcc, mask_flex in train_loader:  # iterate over batches
                    X = X.to(device)
                    Y_struct_true_unmasked = Y_struct.to(device)
                    Y_solvAcc_true_unmasked=Y_solvAcc.to(device)
                    Y_flex_true_unmasked = Y_flex.to(device)
                    mask_struct = mask_struct.to(device)
                    mask_solvAcc = mask_solvAcc.to(device)
                    mask_flex = mask_flex.to(device)

                    Y_struct_raw, Y_solvAcc_raw, Y_flex_raw = model(X)
                    loss_struct = crits[0](Y_struct_raw, Y_struct_true_unmasked)
                    loss_struct *= mask_struct
                    loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
                    total_losses_train[0] += loss_struct.item()

                    loss_solvAcc = crits[1](Y_solvAcc_raw, Y_solvAcc_true_unmasked)
                    loss_solvAcc *= mask_solvAcc
                    loss_solvAcc = loss_solvAcc.sum() / float( mask_solvAcc.sum())  # averages the loss over the structure sequences
                    total_losses_train[1] += loss_solvAcc.item()

                    loss_flex = crits[2](Y_flex_raw, Y_flex_true_unmasked)
                    loss_flex *= mask_flex
                    if(mask_flex.sum()!=0):
                        loss_flex = loss_flex.sum() / float(mask_flex.sum())  # averages the loss over the structure sequences
                        total_losses_train[2] += loss_flex.item()
                        total_losses_train[3] += loss_solvAcc.item() + loss_struct.item() + loss_flex.item()

                    else:
                        total_losses_train[3] += loss_solvAcc.item() + loss_struct.item()


                    Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data,
                                                   dim=1)  # returns index of output predicted with highest probability
                    Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_struct_true = Y_struct_true_unmasked[mask_struct != 0]  # applies weighted mask
                    Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
                    Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

                    Y_struct_true_all = np.append(Y_struct_true_all, Y_struct_true)
                    Y_struct_pred_all = np.append(Y_struct_pred_all, Y_struct_pred)

                avg_losses_train[0] = (total_losses_train[0] / len(train_loader))
                avg_losses_train[1] = (total_losses_train[1] / len(train_loader))
                avg_losses_train[2] = (total_losses_train[2]/ len(train_loader))
                avg_losses_train[3] = (total_losses_train[3]/ len(train_loader))  # avg loss over residues

            return avg_losses_train, Y_struct_true_all, Y_struct_pred_all

        avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train=evaluateTrainLoss()


        if(final_flag): # plotting of bootstrapping in last epoch
            print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
            utilities.plot_bootstrapping(LOG_PATH, bootstrapped_conf_mat)
        #else:
        #    utilities_MICHAEL.add_progress(eval_summary, 'Test', avg_losses_test, epoch,
        #               confusion_matrix, Y_struct_true_all, Y_struct_pred_all)

    return avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train,avg_losses_test, Y_struct_true_all, Y_struct_pred_all, confusion_matrix

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
DSSP_MODE=3       #3 or 8
N_SPLITS_KFOLD=3
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_multitarget'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
#ASA_PATH='/home/mheinzinger/contact_prediction_v2/targets/structured_arrays/asa.npz'

stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=7
LEARNING_RATE= 1e-2

weights=stats_dict[str('proportions'+str(DSSP_MODE))]
weights=weights/float(np.sum(weights))
WEIGHTS_STRUCT=1/weights

WEIGHTS=[WEIGHTS_STRUCT]
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
torch.manual_seed(1234)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex,WEIGHTS)
data_loaders = DataLoader.createDataLoaders(train_set,test_set, 32, 100)

#train_set, test_set=DataLoader.train_val_test_split_KFold(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex,WEIGHTS)
#print('LENGT OF TRAINING SET: ', len(train_set))

#X=np.arange(len(train_set))
#kf=KFold(n_splits=N_SPLITS_KFOLD)
#kf.get_n_splits(X)

overall_train_losses = []
overall_test_losses = []
overall_train_accuracy=[]
overall_test_accuracy=[]

model=ConvNets.MultiTargetNet(INPUT_MODE).to(device)
#model=ConvNets.MultiTargetDenseNet(INPUT_MODE).to(device)

criterion_struct=nn.NLLLoss(reduce=False) #secondary structure
criterion_solvAcc=nn.MSELoss(reduce=False) #solvAcc
criterion_flex=nn.MSELoss(reduce=False) #flex
criterions=[criterion_struct, criterion_solvAcc, criterion_flex]

opt_struct=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
opt_solvAcc=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, amsgrad=True)
opt_flex=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, amsgrad=True)
opts=[opt_struct, opt_solvAcc, opt_flex]

#for train_index, test_index in kf.split(X):
#    X_train =torch.utils.data.Subset(train_set,train_index)
#    X_test=torch.utils.data.Subset(train_set, test_index)
#    data_loaders = DataLoader.createDataLoaders(X_train, X_test, 32, 100)
#einrucken

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
    print('\n')
    print('Epoch '+str(epoch))
    trainNet(model, opts,criterions, data_loaders['Train'])
    train_loss_epoch, train_true_all, train_pred_all, test_loss_epoch, test_true_all,test_pred_all, confusion_matrices_epoch=testNet(model, data_loaders, criterions, epoch, eval_summary, final_flag)

    train_loss_total.append(train_loss_epoch)
    test_loss_total.append(test_loss_epoch)

    #print(confusion_matrices_epoch)

    #if(epoch==0):
    #    confusion_matrices_total_sum.append(confusion_matrices_epoch[0])
    #    confusion_matrices_total_sum.append(confusion_matrices_epoch[1])
    #    confusion_matrices_total_sum.append(confusion_matrices_epoch[2])
    #else:
    #    confusion_matrices_total_sum[0]=confusion_matrices_total_sum[0]+confusion_matrices_epoch[0]
    #    confusion_matrices_total_sum[1] = confusion_matrices_total_sum[1] + confusion_matrices_epoch[1]
    #    confusion_matrices_total_sum[2] = confusion_matrices_total_sum[2] + confusion_matrices_epoch[2]


    print('acc score structure pred:',round(accuracy_score( train_true_all, train_pred_all ),2),round(accuracy_score( test_true_all, test_pred_all ),2))
    train_acc.append(accuracy_score( train_true_all, train_pred_all ))
    test_acc.append(accuracy_score( test_true_all, test_pred_all ))

    if(final_flag):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(np.arange(400), train_true_all[1][:400], color='red')
        plt.plot(np.arange(400), train_pred_all[1][:400], color='blue')
        axes = plt.gca()
        axes.set_ylim([-1, 2])
        plt.savefig('testSolvAccSample.pdf')
#einrcken over
#overall_train_losses.append(train_loss_total)
#overall_test_losses.append(test_loss_total)
#overall_train_accuracy.append(train_acc)
#overall_test_accuracy.append(test_acc)

#average_train_losses=np.average(overall_train_losses, axis=0)
#average_test_losses = np.average(overall_test_losses, axis=0)
#average_train_accuracy = np.average(overall_train_accuracy, axis=0)
#average_test_accuracy = np.average(overall_test_accuracy, axis=0)



utilities.plot_loss_multitask(LOG_PATH, train_loss_total, test_loss_total, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,train_acc, test_acc, NUM_EPOCHS)
utilities.plot_confmat(LOG_PATH, confusion_matrices_epoch )
utilities.get_other_scores(confusion_matrices_epoch)
print('PARAMETERS IN MODEL:', utilities.count_parameters(model))

torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')


















