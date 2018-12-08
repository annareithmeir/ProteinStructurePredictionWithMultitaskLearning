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

    for i, (X, Y_true, mask) in enumerate(train_loader): # iterate over batches
        X = X.to(device)
        Y_true = Y_true.to(device)
        mask=mask.to(device)

        opt.zero_grad()
        Y_raw=model(X)
        loss=crit(Y_raw, Y_true)
        loss*=mask
        loss = loss.sum() / float(mask.sum()) # averages the loss over the structure sequences
        total_loss_train += loss.item()
        Y_pred_unmasked = torch.argmax(Y_raw.data,dim=1)  # returns index of output predicted with highest probability
        Y_pred = Y_pred_unmasked[mask != 0] # applies mask with weights ( inverse to percentage of class)
        Y_true = Y_true[mask != 0]


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
        confusion_matrix=np.zeros((3,3), dtype=np.int) # confusion matrix

        bootstrapped_conf_mat=[] # collects confusion matrices for each sample in last epoch

        for X, Y, mask in test_loader: # iterate over batches
            X = X.to(device)
            Y_true_unmasked = Y.to(device)
            mask = mask.to(device)

            Y_computed = model(X)
            loss=crit(Y_computed, Y_true_unmasked)
            loss*=mask
            loss = (loss.sum()) / float(mask.sum())
            total_loss_test+=loss.item()
            Y_pred_unmasked = torch.argmax(Y_computed.data, dim=1)  # returns index of output predicted with highest probability
            Y_pred = Y_pred_unmasked[mask != 0] # applies weighted mask
            Y_true = Y_true_unmasked[mask != 0] # applies weighted mask

            if (final_flag): # last epoch, bootstrapping
                for i in range(Y_true_unmasked.size()[0]):
                    mask_sample=mask[i]
                    Y_true_sample=Y_true_unmasked[i]
                    Y_pred_sample=Y_pred_unmasked[i]
                    Y_true_sample=Y_true_sample[mask_sample!=0] # applies weighted mask
                    Y_pred_sample=Y_pred_sample[mask_sample!=0] # applies weighted mask
                    #print(mask_sample.size(), Y_true_sample.size(), Y_pred_sample.size())
                    confusion_matrix_per_sample = np.zeros((3, 3), dtype=np.int)
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

                for X, Y, mask in train_loader:  # iterate over batches
                    X = X.to(device)
                    Y_true_unmasked = Y.to(device)
                    mask = mask.to(device)

                    Y_computed = model(X)
                    loss = crit(Y_computed, Y_true_unmasked)
                    loss *= mask
                    loss = (loss.sum()) / float(mask.sum())
                    total_loss_train += loss.item()
                    Y_pred_unmasked = torch.argmax(Y_computed.data,
                                                   dim=1)  # returns index of output predicted with highest probability
                    Y_pred = Y_pred_unmasked[mask != 0]  # applies weighted mask
                    Y_true = Y_true_unmasked[mask != 0]  # applies weighted mask
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

def getData(protein, inputs, targets, masks, input_type, class_type):
    seq=np.load(INPUT_PATH+'/'+protein+'/'+input_type+'.npy')
    tmp={protein:seq}
    inputs.update(tmp)

    struct = np.load(INPUT_PATH + '/' + protein + '/structures_' + str(class_type) + '.npy')
    tmp = {protein: struct}
    targets.update(tmp)

    mask = np.load(INPUT_PATH + '/' + protein + '/mask_' + str(class_type) + '.npy')
    tmp = {protein: mask}
    masks.update(tmp)





INPUT_MODE='protvec_evolutionary'  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3        #3 or 8
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1_7_15_relu'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=100
LEARNING_RATE= 1e-3

weights=stats_dict['proportions3']
WEIGHTS=weights/float(np.sum(weights))
WEIGHTS=1/WEIGHTS
print('WEIGHTS: ',WEIGHTS)

inputs=dict()
targets=dict()
masks=dict()
for prot in os.listdir(INPUT_PATH):
    if (len(os.listdir('preprocessing/' + prot)) == 7):
        getData(prot, inputs, targets, masks,INPUT_MODE, DSSP_MODE)

print('Data length:', len(inputs))
print('Targets length:', len(targets))
print('Masks length:', len(masks))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

train_set, test_set=DataLoader.train_test_split(inputs, targets, masks, WEIGHTS)
print('train:', len(train_set))
print('test:', len(test_set))

data_loaders=DataLoader.createDataLoaders(train_set, test_set, 32, 100)


model=ConvNets.ConvNet_1_7_15(INPUT_MODE).to(device)

criterion=nn.NLLLoss(reduce=False) #reduce=False with mask
#criterion=nn.CrossEntropyLoss(reduce=False)
opt=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

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
    trainNet(model, opt, criterion, data_loaders['Train'])
    train_loss_epoch, train_true_all, train_pred_all, test_loss_epoch, test_true_all,test_pred_all, confusion_matrix_epoch=testNet(model, data_loaders, criterion, epoch, eval_summary, final_flag)

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







