import numpy as np
import os
import time
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


DSSP8_TO_DSSP3_MAPPING = { 0 : 1,
                           1 : 2,
                           2 : 1,
                           3 : 0,
                           4 : 0,
                           5 : 1,
                           6 : 2,
                           7 : 0
                          }


INPUT_MODE=sys.argv[1]  #protvec or onehot or protvec_evolutionary
DSSP_MODE=3       #3 or 8
N_SPLITS_KFOLD=10
LOG_PATH='log/MultiTask/'+str(DSSP_MODE)+'/CNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_multitask'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
#ASA_PATH='/home/mheinzinger/contact_prediction_v2/targets/structured_arrays/asa.npz'

stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

initialize model after each traintestrun!

#Hyperparameters
NUM_EPOCHS=int(sys.argv[2])
LEARNING_RATE= 1e-2

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=Networks.MultiCNN(INPUT_MODE).to(device)

criterion_struct=nn.NLLLoss(reduction='none') #secondary structure
criterion_solvAcc=nn.MSELoss(reduction='none') #solvAcc
criterion_flex=nn.MSELoss(reduction='none') #flex
criterions=[criterion_struct, criterion_solvAcc, criterion_flex]

opt=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

pipeline=MultiTargetPredictionPipeline.Pipeline(DSSP_MODE, device, model, opt, criterions, False, True)


for prot in os.listdir(INPUT_PATH):
    if (len(os.listdir('preprocessing/' + prot)) >= 7 and os.path.exists('preprocessing/'+prot+'/protvec+scoringmatrix.npy')): #only if all input representations available
        pipeline.getData(INPUT_PATH, TARGETS_PATH, prot, inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc, masks_flex,
                INPUT_MODE, DSSP_MODE)

assert(len(inputs)==len(masks_struct)==len(targets_structure)==len(targets_solvAcc)==len(targets_flexibility))
print('NUMBER OF SAMPLES ALLTOGETHER AVAILABLE: ',len(inputs))

print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)
print('SAMPLES:', len(inputs),' OF SIZE ',inputs[inputs.keys()[0]].shape)
print('PARAMETERS IN MODEL:', utilities.count_parameters(model))

utilities.writeNumberOfParameters('compareNumberOfParameters.txt',LOG_PATH, utilities.count_parameters(model))

torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )
utilities.writeLogFile(LOG_PATH, LEARNING_RATE, DSSP_MODE, model)

#train_set, test_set=DataLoader.train_val_test_split(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex,WEIGHTS)
train_set, test_set=DataLoader.train_val_test_split_KFold(inputs, targets_structure, targets_solvAcc, targets_flexibility, masks_struct, masks_solvAcc,masks_flex,WEIGHTS)
#data_loaders = DataLoader.createDataLoaders(train_set,test_set, 32, 100)

X=np.arange(len(train_set))
kf=KFold(n_splits=N_SPLITS_KFOLD)
kf.get_n_splits(X)

overall_train_losses = []
overall_test_losses = []
overall_train_accuracy=[]
overall_test_accuracy=[]

for train_index, test_index in kf.split(X):
    X_train =torch.utils.data.Subset(train_set,train_index)
    X_test=torch.utils.data.Subset(train_set, test_index)
    data_loaders = DataLoader.createDataLoaders(X_train, X_test, 32, 100)

    final_flag=False

    train_loss_total=[]
    test_loss_total=[]
    train_acc=[]
    test_acc=[]
    confusion_matrices_total_sum = []

    START_TIME=time.time()

    for epoch in range(NUM_EPOCHS):
        if(epoch==NUM_EPOCHS-1):
            final_flag=True
        print('\n')
        print('Epoch '+str(epoch))
        pipeline.trainNet(data_loaders['Train'])
        train_loss_epoch, train_true_all, train_pred_all, test_loss_epoch, test_true_all,test_pred_all, confusion_matrices_epoch=pipeline.testNet(data_loaders, final_flag, LOG_PATH)

        train_loss_total.append(train_loss_epoch)
        test_loss_total.append(test_loss_epoch)

        print(confusion_matrices_epoch)

        if(epoch==0):
            confusion_matrices_total_sum.append(confusion_matrices_epoch)
        else:
            confusion_matrices_total_sum=confusion_matrices_total_sum+confusion_matrices_epoch

        if (DSSP_MODE == 8):
            train_pred_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in train_pred_all]
            train_true_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in train_true_all]
            test_pred_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in test_pred_all]
            test_true_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in test_true_all]


        print('TRAIN LOSS:',round(train_loss_epoch[0], 3), round(train_loss_epoch[1], 3),round(train_loss_epoch[2], 3))
        print('TEST LOSS:', round(test_loss_epoch[0], 3),round(test_loss_epoch[1], 3),round(test_loss_epoch[2], 3))
        print('acc score structure pred:',round(accuracy_score( train_true_all, train_pred_all ),2),round(accuracy_score( test_true_all, test_pred_all ),2))
        train_acc.append(round(accuracy_score( train_true_all, train_pred_all ),4))
        test_acc.append(round(accuracy_score( test_true_all, test_pred_all ),4))
        print(train_acc[-1], test_acc[-1])

        utilities.addProgressToLogFile(LOG_PATH, epoch, train_loss_epoch, test_loss_epoch, train_acc[-1], test_acc[-1])

    overall_train_losses.append(train_loss_total)
    overall_test_losses.append(test_loss_total)
    overall_train_accuracy.append(train_acc)
    overall_test_accuracy.append(test_acc)

print(overall_train_accuracy)

END_TIME=time.time()
print('Elapsed time: ', END_TIME-START_TIME)
utilities.writeElapsedTime(LOG_PATH, END_TIME-START_TIME)

average_train_losses=np.average(overall_train_losses, axis=0)
average_test_losses = np.average(overall_test_losses, axis=0)
average_train_accuracy = np.average(overall_train_accuracy, axis=0)
average_test_accuracy = np.average(overall_test_accuracy, axis=0)

utilities.plot_loss_multitask(LOG_PATH, average_train_losses, average_test_losses, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,average_train_accuracy, average_test_accuracy, NUM_EPOCHS)
print(confusion_matrices_total_sum)
utilities.plot_confmat(LOG_PATH, confusion_matrices_total_sum[0] )
utilities.get_other_scores(confusion_matrices_epoch)

torch.save(model.state_dict(), LOG_PATH+'/cnn_multitask.ckpt')