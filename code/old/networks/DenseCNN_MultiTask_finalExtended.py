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
DSSP_MODE=int(sys.argv[3])       #3 or 8
MAP_TO_DSSP3=sys.argv[4]
MULTITASK_MODE=sys.argv[5] # struct, multi2, multi3
assert(MULTITASK_MODE=='struct' or MULTITASK_MODE=='multi2' or MULTITASK_MODE=='multi3'or MULTITASK_MODE=='multi4')
if(DSSP_MODE==8 and MAP_TO_DSSP3=='True'):
    LOG_PATH='log/'+MULTITASK_MODE+'/'+str(DSSP_MODE)+'_mapped/DenseCNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_multitarget'
else:
    LOG_PATH = 'log/'+MULTITASK_MODE+'/' + str(DSSP_MODE) + '/DenseCNN_' + INPUT_MODE + '_' + str(DSSP_MODE) + '_multitarget'
STATS_PATH='stats'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'
#ASA_PATH='/home/mheinzinger/contact_prediction_v2/targets/structured_arrays/asa.npz'

stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

#Hyperparameters
NUM_EPOCHS=int(sys.argv[2])
LEARNING_RATE= 1e-4

weights_struct=stats_dict['proportions'+str(DSSP_MODE)]
weights_struct=weights_struct/float(np.sum(weights_struct))
weights_struct=1/weights_struct
weights_SolvAcc=[1,1]
weights_flex=[1,1,1]
WEIGHTS=[weights_struct, weights_SolvAcc, weights_flex]
print('WEIGHTS:', WEIGHTS)


inputs=dict()
targets_structure3=dict()
targets_structure8=dict()
targets_solvAcc=dict()
targets_flexibility=dict()
masks_struct3=dict()
masks_struct8=dict()
masks_solvAcc=dict()
masks_flex=dict()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=Networks.DenseNet(INPUT_MODE, DSSP_MODE, MULTITASK_MODE).to(device)

criterion_struct=nn.NLLLoss(reduction='none') #secondary structure
criterion_solvAcc=nn.MSELoss(reduction='none') #solvAcc
criterion_flex=nn.MSELoss(reduction='none') #flex
criterions=[criterion_struct, criterion_solvAcc, criterion_flex]

opt=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

pipeline=MultiTargetPredictionPipeline.Pipeline(DSSP_MODE, device, model, opt, criterions, False, MULTITASK_MODE)


for prot in os.listdir(INPUT_PATH):
    if (len(os.listdir('preprocessing/' + prot)) >= 7 and os.path.exists('preprocessing/'+prot+'/protvec+scoringmatrix.npy')): #only if all input representations available
        pipeline.getData(INPUT_PATH, TARGETS_PATH, prot, inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc, masks_flex,
                INPUT_MODE, DSSP_MODE)

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

train_set, test_set, eval_set=DataLoader.train_val_test_split(inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc,masks_flex,WEIGHTS, DSSP_MODE)
data_loaders = DataLoader.createDataLoaders(train_set,test_set,eval_set, 32, 100)

final_flag=False

train_loss_total=[]
test_loss_total=[]
train_acc=[]
test_acc=[]

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

    if (DSSP_MODE == 8 and MAP_TO_DSSP3=='True'):
        train_pred_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in train_pred_all]
        train_true_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in train_true_all]
        test_pred_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in test_pred_all]
        test_true_all = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in test_true_all]

    print('TRAIN LOSS:',round(train_loss_epoch[0], 3), round(train_loss_epoch[1], 3),round(train_loss_epoch[2], 3))
    print('TEST LOSS:', round(test_loss_epoch[0], 3),round(test_loss_epoch[1], 3),round(test_loss_epoch[2], 3))
    print('acc score structure pred:',round(accuracy_score( train_true_all, train_pred_all ),2),round(accuracy_score( test_true_all, test_pred_all ),2))

    train_acc.append(round(accuracy_score( train_true_all, train_pred_all ),3))
    test_acc.append(round(accuracy_score( test_true_all, test_pred_all ),3))

    utilities.addProgressToLogFile(LOG_PATH, epoch, train_loss_epoch, test_loss_epoch, train_acc[-1], test_acc[-1])

END_TIME=time.time()
print('Elapsed time: ', END_TIME-START_TIME)
utilities.writeElapsedTime(LOG_PATH, END_TIME-START_TIME)

utilities.plot_loss_multitask(LOG_PATH, train_loss_total, test_loss_total, NUM_EPOCHS)
utilities.plot_accuracy(LOG_PATH,train_acc, test_acc, NUM_EPOCHS)
utilities.plot_confmat(LOG_PATH, confusion_matrices_epoch )
utilities.get_other_scores(confusion_matrices_epoch)

torch.save(model.state_dict(), LOG_PATH+'/dense_cnn_multitask.ckpt')