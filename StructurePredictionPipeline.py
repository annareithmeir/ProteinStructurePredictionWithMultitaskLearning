import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities_MICHAEL as utilities
from sklearn.metrics import accuracy_score

import ConvNets
import DataLoader

def trainNet(model, opt, crit, train_loader):
    model.train()
    total_loss_train = 0
    Y_true_all = np.zeros(0, dtype=np.int)
    Y_pred_all = np.zeros(0, dtype=np.int)
    c=0
    w=0
    for i, (X, Y_true, mask) in enumerate(train_loader):
        X = X.to(device)
        Y_true = Y_true.to(device)
        mask=mask.to(device)

        opt.zero_grad()
        Y_raw=model(X)
        loss=crit(Y_raw, Y_true)
        loss*=mask
        loss = loss.sum() / float(mask.sum())
        total_loss_train += loss.item()
        Y_pred_unmasked = torch.argmax(Y_raw.data,dim=1)  # returns index of output predicted with highest probability
        #Y_pred = Y_pred_unmasked[mask == 1]
        #Y_true = Y_true[mask == 1]
        Y_pred = Y_pred_unmasked[mask != 0]
        Y_true = Y_true[mask != 0]

        '''
        for i in range(mask.size()[2]):
            if (mask[0][0][i] == 1):
                if (Y_pred[i] == Y_true[i]):
                    c += 1
                else:
                    w += 1
        '''

        Y_true = Y_true.view(-1).long().cpu().numpy()
        Y_pred = Y_pred.view(-1).long().cpu().numpy()

        Y_true_all = np.append(Y_true_all, Y_true)
        Y_pred_all = np.append(Y_pred_all, Y_pred)

        loss.backward()
        opt.step()
    #accuracy = (c / float(c + w)) * 100
    total_loss_train=total_loss_train / len(train_loader)
    print('--', len(train_loader),total_loss_train)
    return total_loss_train, Y_true_all, Y_pred_all

def testNet(model, test_loader, crit, epoch, eval_summary, final_flag):
    model.eval()
    with torch.no_grad():
        total_loss_test=0
        bootstrapping=-1
        Y_true_all = np.zeros(0, dtype=np.int)
        Y_pred_all = np.zeros(0, dtype=np.int)
        confusion_matrix=np.zeros((3,3), dtype=np.int)

        bootstrapped_conf_mat=[]

        c = 0
        w = 0

        for X, Y, mask in test_loader: #batchsize samples in X and Y
            X = X.to(device)
            Y_true_unmasked = Y.to(device)
            mask = mask.to(device)
            Y_computed = model(X)
            loss=crit(Y_computed, Y_true_unmasked)
            loss*=mask
            loss = (loss.sum()) / float(mask.sum())
            total_loss_test+=loss.item()
            Y_pred_unmasked = torch.argmax(Y_computed.data, dim=1)  # returns index of output predicted with highest probability
            #Y_pred = Y_pred_unmasked[ mask==1 ]
            #Y_true = Y_true_unmasked[ mask==1 ]
            Y_pred = Y_pred_unmasked[mask != 0]
            Y_true = Y_true_unmasked[mask != 0]

            '''
            for i in range(mask.size()[2]):
                if(mask[0][0][i]==1):
                    if (Y_pred[i] == Y_true[i]):
                        c += 1
                    else:
                        w += 1
            '''
            if (final_flag):
                for i in range(Y_true_unmasked.size()[0]):
                    mask_sample=mask[i]
                    Y_true_sample=Y_true_unmasked[i]
                    Y_pred_sample=Y_pred_unmasked[i]
                    Y_true_sample=Y_true_sample[mask_sample!=0]
                    Y_pred_sample=Y_pred_sample[mask_sample!=0]
                    #print(mask_sample.size(), Y_true_sample.size(), Y_pred_sample.size())
                    confusion_matrix_per_sample = np.zeros((3, 3), dtype=np.int)
                    np.add.at(confusion_matrix_per_sample, (Y_true_sample, Y_pred_sample), 1)
                    print(confusion_matrix_per_sample)
                    bootstrapped_conf_mat.insert(0, confusion_matrix_per_sample)

            #percentage_correct.insert(0,c / float(c + w))
            Y_true = Y_true.view(-1).long().cpu().numpy()
            Y_pred = Y_pred.view(-1).long().cpu().numpy()
            np.add.at(confusion_matrix, (Y_true, Y_pred), 1)

            Y_true_all = np.append(Y_true_all, Y_true)
            Y_pred_all = np.append(Y_pred_all, Y_pred)



        total_loss_test = (total_loss_test / len(test_loader))

        if(final_flag):
            #print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
            #utilities.add_progress(eval_summary, 'Test', total_loss, epoch,
            #                       confusion_matrix, Y_true_all, Y_pred_all, bootstrapped_conf_mat)
            #print(bootstrapped_conf_mat)
            utilities.plot_bootstrapping(LOG_PATH, bootstrapped_conf_mat)
        #else:
        #utilities.add_progress(eval_summary, 'Test', total_loss, epoch,
        #               confusion_matrix, Y_true_all, Y_pred_all)


    #accuracy=(c/float(c+w))*100
    return total_loss_test, Y_true_all, Y_pred_all, confusion_matrix


INPUT_MODE='1hot'  #protvec or 1hot
DSSP_MODE=3        #3 or 8
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1_7_15_relu'
STATS_PATH='stats'
MATRIX_PATH='matrices_012'
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

NUM_EPOCHS=100
LEARNING_RATE= 1e-3

inputs = np.load(MATRIX_PATH+'/matrix_'+INPUT_MODE+'_train.npy', fix_imports=True)
targets=np.load(MATRIX_PATH+'/targets_'+str(DSSP_MODE)+'_train.npy', fix_imports=True)
masks=np.load(MATRIX_PATH+'/masks_'+str(DSSP_MODE)+'_train.npy', fix_imports=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', inputs.shape,' OF SIZE ',inputs[0].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )

for i in range(len(inputs)):
    assert not np.any(np.isnan(inputs[i]))
    assert not np.any(np.isnan(targets[i]))

weights=stats_dict['proportions3']
print(weights)
print(np.sum(weights))
WEIGHTS=weights/float(np.sum(weights))
print('WEIGHTS: ',WEIGHTS)

train_set, test_set=DataLoader.train_test_split(inputs, targets, masks, WEIGHTS)
data_loaders=DataLoader.createDataLoaders(train_set, test_set, 32, 100)

model=ConvNets.ConvNet_1_7_15(INPUT_MODE).to(device)



#criterion=nn.NLLLoss(reduce=False) #reduce=False with mask
criterion=nn.CrossEntropyLoss(reduce=False)
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
    train_loss_epoch, train_true_all, train_pred_all=trainNet(model, opt, criterion, data_loaders['Train'])
    test_loss_epoch, test_true_all,test_pred_all, confusion_matrix_epoch=testNet(model, data_loaders['Test'], criterion, epoch, eval_summary, final_flag)
    print('TRAIN LOSS:', train_loss_epoch)
    print('TEST LOSS:',test_loss_epoch)
    train_loss.append(train_loss_epoch)
    test_loss.append(test_loss_epoch)
    print(train_pred_all, train_true_all)
    print('acc score:',accuracy_score( train_true_all, train_pred_all ),accuracy_score( test_true_all, test_pred_all ))
    train_acc.append(accuracy_score( train_true_all, train_pred_all ))
    test_acc.append(accuracy_score( test_true_all, test_pred_all ))
    #utilities.save_performance(LOG_PATH, eval_summary)
    #utilities.plot_learning(LOG_PATH, eval_summary)

    '''
    if (epoch % 10) == 0: #superconvergence
        batch_size=batch_size+8
        data_loaders = DataLoader.createDataLoaders(train_set, test_set, batch_size)
    '''


#utilities.plot_learning( LOG_PATH, eval_summary)
#utilities.save_performance( LOG_PATH, eval_summary, final_flag=True )

utilities.plot_loss(LOG_PATH, train_loss, test_loss, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,train_acc, test_acc, NUM_EPOCHS)
utilities.plot_confmat(LOG_PATH, confusion_matrix_epoch )
utilities.get_other_scores(confusion_matrix_epoch)

torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')






