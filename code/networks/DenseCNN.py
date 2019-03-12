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

#
# This file is the DenseCNN model and takes the following arguments when being called:
# 1. Input mode (1hot, protvec, protvecevolutionary, pssm, protvec+scoringmatrix)
# 2. Number of epochs
# 3. DSSP classification mode (3,8)
# 4. Map to DSSP3? (True or False, no effect on DSSP3 classification)
# 5. Multi-task mode .struct (DSSP), multi2 (DSSP, RSA), multi (DSSP, RSA, B-factors), multi4 (DSSP3, DSSP8, RSA, B-factors)
#
#
# Note: With multi4, the dssp-mode determines which of the two dssp types is mainly learned. The implementation allows both.
#

DSSP8_TO_DSSP3_MAPPING = { 0 : 1,
                           1 : 2,
                           2 : 1,
                           3 : 0,
                           4 : 0,
                           5 : 1,
                           6 : 2,
                           7 : 0
                          }


INPUT_MODE=sys.argv[1]
NUM_EPOCHS=int(sys.argv[2])
DSSP_MODE=int(sys.argv[3])
MAP_TO_DSSP3=sys.argv[4]
MULTITASK_MODE=sys.argv[5]
assert(MULTITASK_MODE=='struct' or MULTITASK_MODE=='multi2' or MULTITASK_MODE=='multi3'or MULTITASK_MODE=='multi4')
if(DSSP_MODE==8 and MAP_TO_DSSP3=='True'):
    LOG_PATH='log/'+MULTITASK_MODE+'/'+str(DSSP_MODE)+'_mapped/DenseCNN_'+INPUT_MODE+'_'+str(DSSP_MODE)+'_multitask'
else:
    LOG_PATH = 'log/'+MULTITASK_MODE+'/' + str(DSSP_MODE) + '/DenseCNN_' + INPUT_MODE + '_' + str(DSSP_MODE) + '_multitask'
STATS_PATH='data analysis'
TEST_PATH='testset_preprocessed'
VAL_PATH='validationset_preprocessed'
TRAIN_PATH='trainset_preprocessed'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'

# class distr saved here for weighting
stats_dict=np.load(STATS_PATH+'/stats_dict.npy').item()

LEARNING_RATE= 1e-4

weights_struct=stats_dict['proportions'+str(DSSP_MODE)]
weights_struct=weights_struct/float(np.sum(weights_struct))
weights_struct=1/weights_struct
weights_SolvAcc=[1,1]
weights_flex=[1,1,1]
WEIGHTS=[weights_struct, weights_SolvAcc, weights_flex]

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

criterion_struct=nn.NLLLoss(reduction='none')
criterion_solvAcc=nn.MSELoss(reduction='none')
criterion_flex=nn.MSELoss(reduction='none')
criterions=[criterion_struct, criterion_solvAcc, criterion_flex]

opt=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

pipeline=MultiTargetPredictionPipeline.Pipeline(DSSP_MODE, device, model, opt, criterions, False, MULTITASK_MODE, MAP_TO_DSSP3)


for prot in os.listdir(TRAIN_PATH):
    pipeline.getData(TRAIN_PATH, TARGETS_PATH,prot, inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, INPUT_MODE, DSSP_MODE)

assert(len(inputs)==len(masks_struct3)==len(targets_structure3)==len(masks_struct8)==len(targets_structure8)==len(targets_solvAcc)==len(targets_flexibility))

inputs_test=dict()
targets_structure3_test=dict()
targets_structure8_test=dict()
targets_solvAcc_test=dict()
targets_flexibility_test=dict()
masks_struct3_test=dict()
masks_struct8_test=dict()
masks_solvAcc_test=dict()
masks_flex_test=dict()

for prot in os.listdir(TEST_PATH):
    pipeline.getData(TEST_PATH, TARGETS_PATH,prot, inputs_test, targets_structure3_test, targets_structure8_test, targets_solvAcc_test, targets_flexibility_test, masks_struct3_test, masks_struct8_test, masks_solvAcc_test, masks_flex_test,INPUT_MODE, DSSP_MODE)

assert(len(inputs_test)==len(masks_struct3_test)==len(targets_structure3_test)==len(masks_struct8_test)==len(targets_structure8_test)==len(targets_solvAcc_test)==len(targets_flexibility_test))

print('DEVICE: ',device)
print('INPUT MODE: ', INPUT_MODE)
print('DSSP CLASSIFICATION MODE: ', DSSP_MODE)

utilities.writeNumberOfParameters('compareNumberOfParameters.txt',LOG_PATH, utilities.count_parameters(model))

torch.manual_seed(42)
utilities.create_logdir( LOG_PATH )
utilities.writeLogFile(LOG_PATH, LEARNING_RATE, DSSP_MODE, model)

train_set=DataLoader.make_dataset(inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc,masks_flex,WEIGHTS, DSSP_MODE)
test_set=DataLoader.make_dataset(inputs_test, targets_structure3_test, targets_structure8_test, targets_solvAcc_test, targets_flexibility_test, masks_struct3_test, masks_struct8_test, masks_solvAcc_test,masks_flex_test,WEIGHTS, DSSP_MODE)
data_loaders = DataLoader.createDataLoaders(train_set,test_set, 32, 100)

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

    print('TRAIN LOSS:',round(train_loss_epoch[3], 3))
    print('TEST LOSS:', round(test_loss_epoch[3], 3))
    print('acc score structure pred:',round(accuracy_score( train_true_all, train_pred_all ),2),round(accuracy_score( test_true_all, test_pred_all ),2))

    train_acc.append(round(accuracy_score( train_true_all, train_pred_all ),3))
    test_acc.append(round(accuracy_score( test_true_all, test_pred_all ),3))

    utilities.addProgressToLogFile(LOG_PATH, epoch, train_loss_epoch, test_loss_epoch, train_acc[-1], test_acc[-1])

END_TIME=time.time()
print('Elapsed time: ', END_TIME-START_TIME)
utilities.writeElapsedTime(LOG_PATH, END_TIME-START_TIME)
utilities.write_confmat(LOG_PATH, confusion_matrices_epoch )

torch.save(model.state_dict(), LOG_PATH+'/dense_cnn_multitask.ckpt')