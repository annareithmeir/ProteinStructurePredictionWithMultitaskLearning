import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities_MICHAEL as utilities

import ConvNets
import DataLoader


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


INPUT_MODE='1hot'  #protvec or 1hot
DSSP_MODE=3        #3 or 8
LOG_PATH='log/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_1-7-15_relu'

NUM_EPOCHS=100
LEARNING_RATE=1e-3


inputs = np.load('matrix_'+INPUT_MODE+'_train.npy', fix_imports=True)
targets=np.load('targets_'+str(DSSP_MODE)+'_train.npy', fix_imports=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', inputs.shape,' OF SIZE ',inputs[0].shape)
torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )


train_set, test_set=DataLoader.train_test_split(inputs, targets)
data_loaders=DataLoader.createDataLoaders(train_set, test_set)

model=ConvNets.ConvNet_1_7_15(INPUT_MODE).to(device)
criterion=nn.NLLLoss(reduce=False) #reduce=False with mask
opt=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)


eval_summary = dict()
for epoch in range(NUM_EPOCHS):
    print('\n')
    print('Epoch '+str(epoch))
    trainNet(model, opt, criterion, data_loaders['Train'])
    testNet(model, data_loaders, criterion, epoch, eval_summary)
    utilities.save_performance(LOG_PATH, eval_summary)
    utilities.plot_learning(LOG_PATH, eval_summary)

utilities.plot_learning( LOG_PATH, eval_summary )
utilities.save_performance( LOG_PATH, eval_summary, final_flag=True )

torch.save(model.state_dict(), LOG_PATH+'/model.ckpt')
