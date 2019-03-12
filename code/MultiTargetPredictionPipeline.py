import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from random import randint

#
# This file handles the overall training, testing and validating process.
# All lists (e.g. crits, total_losses,...) are arraged in the following order: [struct, solvAcc, flex, total] or [struct, solvAcc, flex].
#
# In the case of learning all four targets, the dssp_mode determines if dssp3 or dssp8 is considered the main task.
# This implementation allows all possible combinations: 5 inputs x 6 networks x 4 multi-task modes (3 MTL+ only struct)= 120.
# In addition to that, multi4 allows two variants (learning dssp3 mainly or dssp8 mainly). ---> + 30 configurations.
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

class Pipeline():
    def __init__(self, dssp_mode, device, model, opt, crits, lstm_flag, multi_mode, map):
        self.dssp_mode=dssp_mode
        self.device=device
        self.model=model
        self.opt=opt
        self.crits=crits
        self.lstm_flag=lstm_flag
        self.multi_mode=multi_mode
        self.map=map

    #
    # Training step
    #

    def trainNet(self, train_loader):

        self.model.train()
        # iterate over batches
        for i, (X, Y_struct3_true, Y_struct8_true, Y_solvAcc_true, Y_flex_true, mask_struct3, mask_struct8, mask_solvAcc, mask_flex, len_wo_padding) in enumerate(train_loader):

            #Prepare inputs and targets
            X = X.to(self.device)
            if(self.multi_mode=='multi4'):
                #In the case of learning both DSSP types, the targets and masks are organized in a list
                Y_struct_true = [Y_struct3_true.to(self.device),Y_struct8_true.to(self.device)]
                mask_struct = [mask_struct3.to(self.device), mask_struct8.to(self.device)]
            elif(self.dssp_mode==8):
                Y_struct_true = Y_struct8_true.to(self.device)
                mask_struct = mask_struct8.to(self.device)
            else:
                Y_struct_true = Y_struct3_true.to(self.device)
                mask_struct = mask_struct3.to(self.device)

            Y_solvAcc_true = Y_solvAcc_true.to(self.device)
            Y_flex_true = Y_flex_true.to(self.device)

            mask_solvAcc=mask_solvAcc.to(self.device)
            mask_flex = mask_flex.to(self.device)

            #LSTMs need to be reinitialized
            if (self.lstm_flag):
                self.model.zero_grad()
                self.model.hidden = self.model.init_hidden(X.size()[0])
            self.opt.zero_grad()

            #Get predictions
            Y_struct_raw, Y_solvAcc_raw, Y_flex_raw = self.model(X)

            #Depending on the multi-task type, compute total loss of model
            if(self.multi_mode=='multi2'):
                loss_struct = self.crits[0](Y_struct_raw, Y_struct_true)
                loss_struct *= mask_struct
                loss_struct = loss_struct.sum() / float(mask_struct.sum())

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())

                loss = loss_struct + loss_solvAcc

            elif(self.multi_mode=='multi3'):
                loss_struct = self.crits[0](Y_struct_raw, Y_struct_true)
                loss_struct *= mask_struct
                loss_struct = loss_struct.sum() / float(mask_struct.sum())

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())

                loss_flex = self.crits[2](Y_flex_raw, Y_flex_true)
                loss_flex *= mask_flex
                if (mask_flex.sum() != 0):  # we have one sample where the mask is zero everywhere (2EF4)
                    loss_flex = loss_flex.sum() / float(mask_flex.sum())
                else:
                    loss_flex = torch.zeros(1)

                loss = loss_struct + loss_solvAcc +loss_flex

            elif (self.multi_mode == 'multi4'):
                loss_struct3 = self.crits[0](Y_struct_raw[0], Y_struct_true[0])
                loss_struct3 *= mask_struct[0]
                loss_struct3 = loss_struct3.sum() / float(mask_struct[0].sum())

                loss_struct8 = self.crits[0](Y_struct_raw[1], Y_struct_true[1])
                loss_struct8 *= mask_struct[1]
                loss_struct8 = loss_struct8.sum() / float(mask_struct[1].sum())

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(
                    mask_solvAcc.sum())

                loss_flex = self.crits[2](Y_flex_raw, Y_flex_true)
                loss_flex *= mask_flex
                if (mask_flex.sum() != 0):  # we have one sample where the mask is zero everywhere (2EF4)
                    loss_flex = loss_flex.sum() / float(mask_flex.sum())
                else:
                    loss_flex = torch.zeros(1)

                loss = loss_struct3+ loss_struct8 + loss_solvAcc + loss_flex

            else:
                loss_struct = self.crits[0](Y_struct_raw, Y_struct_true)
                loss_struct *= mask_struct
                loss_struct = loss_struct.sum() / float(mask_struct.sum())

                loss = loss_struct

            loss.backward()
            self.opt.step()

    #
    # Testing step
    #

    def testNet(self, data_loaders, final_flag, log_path):
        test_loader=data_loaders['Test']
        train_loader=data_loaders['Train']

        self.model.eval()

        with torch.no_grad():
            total_losses_test=[0,0,0,0] #[struct,solvAcc,flex,sum]
            avg_losses_test=[0,0,0,0] #[struct,solvAcc,flex,sum]
            pred_dict=dict()

            Y_struct_true_accumulate_test = np.zeros(0, dtype=np.int) # collects true targets
            Y_struct_pred_accumulate_test = np.zeros(0, dtype=np.int) # collects predicted targets
            if(self.dssp_mode==8 and self.map=='True'):
                confusion_matrix = np.zeros((3,3), dtype=np.int)  # confusion matrix
            else:
                confusion_matrix=np.zeros((self.dssp_mode,self.dssp_mode), dtype=np.int) # confusion matrix
            confmats_list=[] # collects confusion matrices for each sample in last epoch
            R2_list=[]

            # Evaluates the training set on the already trained model to compare to test set.
            # This time, the masks need to be applied.
            def evaluateTrainLoss():
                with torch.no_grad():
                    total_losses_train=[0,0,0,0] #[struct,solvAcc,flex,sum]
                    avg_losses_train=[0,0,0,0] #[struct,solvAcc,flex,sum]

                    Y_struct_true_accumulate_train = np.zeros(0, dtype=np.int)  # collects true targets
                    Y_struct_pred_accumulate_train = np.zeros(0, dtype=np.int)  # collects predicted targets


                    for X_train, Y_struct3_train,Y_struct8_train, Y_solvAcc_train, Y_flex_train, mask_struct3_train,mask_struct8_train, mask_solvAcc_train, mask_flex_train, len_wo_padding_train in train_loader:  # iterate over batches

                        #Prepare inputs
                        X_train = X_train.to(self.device)
                        if(self.multi_mode=='multi4'):
                            #In the case of learning both DSSP types, the targets and masks are organized in a list
                            Y_struct_true_unmasked_train = [Y_struct3_train.to(self.device), Y_struct8_train.to(self.device)]
                            mask_struct_train = [mask_struct3_train.to(self.device), mask_struct8_train.to(self.device)]
                        elif(self.dssp_mode==8):
                            Y_struct_true_unmasked_train = Y_struct8_train.to(self.device)
                            mask_struct_train = mask_struct8_train.to(self.device)
                        else:
                            Y_struct_true_unmasked_train = Y_struct3_train.to(self.device)
                            mask_struct_train = mask_struct3_train.to(self.device)

                        Y_solvAcc_true_unmasked_train=Y_solvAcc_train.to(self.device)
                        Y_flex_true_unmasked_train = Y_flex_train.to(self.device)
                        mask_solvAcc_train = mask_solvAcc_train.to(self.device)
                        mask_flex_train = mask_flex_train.to(self.device)

                        # LSTMs need to be reinitialized
                        if (self.lstm_flag):
                            self.model.zero_grad()
                            self.model.hidden = self.model.init_hidden(X_train.size()[0])

                        # Get predictions
                        Y_struct_raw_train, Y_solvAcc_raw_train, Y_flex_raw_train = self.model(X_train)

                        #Depending on the multi-task type, compute total loss of model
                        if(self.multi_mode=='multi2'):
                            loss_struct_train = self.crits[0](Y_struct_raw_train, Y_struct_true_unmasked_train)
                            loss_struct_train *= mask_struct_train
                            loss_struct_train = loss_struct_train.sum() / float(mask_struct_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[0] += loss_struct_train.item()

                            loss_solvAcc_train = self.crits[1](Y_solvAcc_raw_train, Y_solvAcc_true_unmasked_train)
                            loss_solvAcc_train *= mask_solvAcc_train
                            loss_solvAcc_train = loss_solvAcc_train.sum() / float( mask_solvAcc_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[1] += loss_solvAcc_train.item()

                            total_losses_train[3] += loss_solvAcc_train.item() + loss_struct_train.item()

                        elif(self.multi_mode=='multi3'):
                            loss_struct_train = self.crits[0](Y_struct_raw_train, Y_struct_true_unmasked_train)
                            loss_struct_train *= mask_struct_train
                            loss_struct_train = loss_struct_train.sum() / float(
                                mask_struct_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[0] += loss_struct_train.item()

                            loss_solvAcc_train = self.crits[1](Y_solvAcc_raw_train, Y_solvAcc_true_unmasked_train)
                            loss_solvAcc_train *= mask_solvAcc_train
                            loss_solvAcc_train = loss_solvAcc_train.sum() / float(
                                mask_solvAcc_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[1] += loss_solvAcc_train.item()

                            loss_flex_train = self.crits[2](Y_flex_raw_train, Y_flex_true_unmasked_train)
                            loss_flex_train *= mask_flex_train
                            if (mask_flex_train.sum() != 0):
                                loss_flex_train = loss_flex_train.sum() / float(
                                    mask_flex_train.sum())  # averages the loss over the structure sequences
                                total_losses_train[2] += loss_flex_train.item()

                                total_losses_train[3] += loss_solvAcc_train.item() + loss_struct_train.item() + loss_flex_train.item()

                            else:
                                total_losses_train[3] += loss_solvAcc_train.item() + loss_struct_train.item()

                        elif (self.multi_mode == 'multi4'):
                            loss_struct3_train = self.crits[0](Y_struct_raw_train[0], Y_struct_true_unmasked_train[0])
                            loss_struct3_train *= mask_struct_train[0]
                            loss_struct3_train = loss_struct3_train.sum() / float(mask_struct_train[0].sum())  # averages the loss over the structure sequences
                            total_losses_train[0] += loss_struct3_train.item()

                            loss_struct8_train = self.crits[0](Y_struct_raw_train[1], Y_struct_true_unmasked_train[1])
                            loss_struct8_train *= mask_struct_train[1]
                            loss_struct8_train = loss_struct8_train.sum() / float( mask_struct_train[1].sum())  # averages the loss over the structure sequences
                            total_losses_train[0] += loss_struct8_train.item()

                            loss_solvAcc_train = self.crits[1](Y_solvAcc_raw_train, Y_solvAcc_true_unmasked_train)
                            loss_solvAcc_train *= mask_solvAcc_train
                            loss_solvAcc_train = loss_solvAcc_train.sum() / float(
                                mask_solvAcc_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[1] += loss_solvAcc_train.item()

                            loss_flex_train = self.crits[2](Y_flex_raw_train, Y_flex_true_unmasked_train)
                            loss_flex_train *= mask_flex_train
                            if (mask_flex_train.sum() != 0):
                                loss_flex_train = loss_flex_train.sum() / float(
                                    mask_flex_train.sum())  # averages the loss over the structure sequences
                                total_losses_train[2] += loss_flex_train.item()

                                total_losses_train[3] += loss_solvAcc_train.item() + loss_struct3_train.item()+ loss_struct8_train.item() + loss_flex_train.item()

                            else:
                                total_losses_train[3] += loss_solvAcc_train.item() + loss_struct3_train.item() + loss_struct8_train.item()

                        else:
                            loss_struct_train = self.crits[0](Y_struct_raw_train, Y_struct_true_unmasked_train)
                            loss_struct_train *= mask_struct_train
                            loss_struct_train = loss_struct_train.sum() / float(
                                mask_struct_train.sum())  # averages the loss over the structure sequences
                            total_losses_train[0] += loss_struct_train.item()

                            total_losses_train[3] += loss_struct_train.item()

                        #If learning all targets simultaneously then DSSP3 and DSSP8 classification is possible, decision needed. In this thesis only DSSP3 is examined with multi-task,
                        #but this code also enables to examine DSSP8 + multi-task.structure  masks and predictions are organized in list [dssp3,dssp8] in multi4.
                        if(self.multi_mode=='multi4'):
                            if(self.dssp_mode==3):
                                Y_struct_pred_unmasked_train = torch.argmax(Y_struct_raw_train[0].data,dim=1)  # returns index of output predicted with highest probability
                                assert (torch.max(Y_struct_pred_unmasked_train) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked_train) >= 0)

                                Y_struct_pred_train = Y_struct_pred_unmasked_train[ mask_struct_train[0] != 0]  # applies weighted mask
                                Y_struct_true_train = Y_struct_true_unmasked_train[0][ mask_struct_train[0] != 0]  # applies weighted mask

                                Y_struct_true_train = Y_struct_true_train.view(-1).long().cpu().numpy()
                                Y_struct_pred_train = Y_struct_pred_train.view(-1).long().cpu().numpy()

                                Y_struct_true_accumulate_train = np.append(Y_struct_true_accumulate_train, Y_struct_true_train)
                                Y_struct_pred_accumulate_train = np.append(Y_struct_pred_accumulate_train,Y_struct_pred_train)

                            else:
                                Y_struct_pred_unmasked_train = torch.argmax(Y_struct_raw_train[1].data, dim=1)  # returns index of output predicted with highest probability
                                assert (torch.max(Y_struct_pred_unmasked_train) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked_train) >= 0)

                                Y_struct_pred_train = Y_struct_pred_unmasked_train[mask_struct_train[1] != 0]  # applies weighted mask
                                Y_struct_true_train = Y_struct_true_unmasked_train[1][ mask_struct_train[1] != 0]  # applies weighted mask

                                Y_struct_true_train = Y_struct_true_train.view(-1).long().cpu().numpy()
                                Y_struct_pred_train = Y_struct_pred_train.view(-1).long().cpu().numpy()

                                Y_struct_true_accumulate_train = np.append(Y_struct_true_accumulate_train, Y_struct_true_train)
                                Y_struct_pred_accumulate_train = np.append(Y_struct_pred_accumulate_train, Y_struct_pred_train)

                        #In all other cases the model determines the DSSPtype already and the structure targets and masks are not organized in a list
                        else:

                            Y_struct_pred_unmasked_train = torch.argmax(Y_struct_raw_train.data, dim=1)  # returns index of output predicted with highest probability
                            assert (torch.max(Y_struct_pred_unmasked_train) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked_train) >= 0)

                            Y_struct_pred_train = Y_struct_pred_unmasked_train[mask_struct_train != 0]  # applies weighted mask
                            Y_struct_true_train = Y_struct_true_unmasked_train[mask_struct_train != 0]  # applies weighted mask

                            Y_struct_true_train = Y_struct_true_train.view(-1).long().cpu().numpy()
                            Y_struct_pred_train = Y_struct_pred_train.view(-1).long().cpu().numpy()


                            Y_struct_true_accumulate_train = np.append(Y_struct_true_accumulate_train, Y_struct_true_train)
                            Y_struct_pred_accumulate_train = np.append(Y_struct_pred_accumulate_train, Y_struct_pred_train)


                    avg_losses_train[0] = round((total_losses_train[0] / len(train_loader)),3)
                    avg_losses_train[1] = round((total_losses_train[1] / len(train_loader)),3)
                    avg_losses_train[2] = round((total_losses_train[2]/ len(train_loader)),3)
                    avg_losses_train[3] = round((total_losses_train[3]/ len(train_loader)),3)  # avg loss over residues

                return avg_losses_train, Y_struct_true_accumulate_train, Y_struct_pred_accumulate_train



            #First, evaluate training set
            avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train=evaluateTrainLoss()

            #Then, evaluate test/validation set
            for X, Y_struct3_true,Y_struct8_true, Y_solvAcc_true, Y_flex_true, mask_struct3, mask_struct8, mask_solvAcc, mask_flex, len_wo_padding in test_loader: # iterate over batches

                #X = torch.rand(X.size()[0], X.size()[1], X.size()[2], X.size()[3])

                #Prepare inputs and targets
                X = X.to(self.device)
                if (self.multi_mode == 'multi4'):
                    Y_struct_true_unmasked = [Y_struct3_true.to(self.device), Y_struct8_true.to(self.device)]
                    mask_struct = [mask_struct3.to(self.device), mask_struct8.to(self.device)]
                elif(self.dssp_mode==8):
                    Y_struct_true_unmasked = Y_struct8_true.to(self.device)
                    mask_struct = mask_struct8.to(self.device)
                else:
                    Y_struct_true_unmasked = Y_struct3_true.to(self.device)
                    mask_struct = mask_struct3.to(self.device)

                Y_solvAcc_true_unmasked = Y_solvAcc_true.to(self.device)
                Y_flex_true_unmasked = Y_flex_true.to(self.device)
                mask_solvAcc=mask_solvAcc.to(self.device)
                mask_flex = mask_flex.to(self.device)

                #LSTMs need to be reinitialized
                if (self.lstm_flag):
                    self.model.zero_grad()
                    self.model.hidden = self.model.init_hidden(X.size()[0])

                #Get predictions
                Y_struct_raw,  Y_solvAcc_raw, Y_flex_raw = self.model(X)

                #Depending on the multi-task type the loss is computed
                if(self.multi_mode=='multi2'):
                    loss_struct = self.crits[0](Y_struct_raw, Y_struct_true_unmasked)
                    loss_struct *= mask_struct
                    loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
                    total_losses_test[0] += loss_struct.item()

                    loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true_unmasked)
                    loss_solvAcc *= mask_solvAcc
                    loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences
                    total_losses_test[1] += loss_solvAcc.item()

                    total_losses_test[3] += loss_solvAcc.item() + loss_struct.item()

                elif (self.multi_mode == 'multi3'):
                    loss_struct = self.crits[0](Y_struct_raw, Y_struct_true_unmasked)
                    loss_struct *= mask_struct
                    loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
                    total_losses_test[0] += loss_struct.item()

                    loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true_unmasked)
                    loss_solvAcc *= mask_solvAcc
                    loss_solvAcc = loss_solvAcc.sum() / float(
                        mask_solvAcc.sum())  # averages the loss over the structure sequences
                    total_losses_test[1] += loss_solvAcc.item()

                    loss_flex = self.crits[2](Y_flex_raw, Y_flex_true_unmasked)
                    loss_flex *= mask_flex
                    loss_flex = loss_flex.sum() / float(
                        mask_flex.sum())  # averages the loss over the structure sequences
                    total_losses_test[2] += loss_flex.item()

                    total_losses_test[3] += loss_solvAcc.item() + loss_struct.item() + loss_flex.item()

                elif (self.multi_mode == 'multi4'):
                    loss_struct3 = self.crits[0](Y_struct_raw[0], Y_struct_true_unmasked[0])
                    loss_struct3 *= mask_struct[0]
                    loss_struct3 = (loss_struct3.sum()) / float(mask_struct[0].sum())
                    total_losses_test[0] += loss_struct3.item()

                    loss_struct8 = self.crits[0](Y_struct_raw[1], Y_struct_true_unmasked[1])
                    loss_struct8 *= mask_struct[1]
                    loss_struct8 = (loss_struct8.sum()) / float(mask_struct[1].sum())
                    total_losses_test[0] += loss_struct8.item()

                    loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true_unmasked)
                    loss_solvAcc *= mask_solvAcc
                    loss_solvAcc = loss_solvAcc.sum() / float(
                        mask_solvAcc.sum())  # averages the loss over the structure sequences
                    total_losses_test[1] += loss_solvAcc.item()

                    loss_flex = self.crits[2](Y_flex_raw, Y_flex_true_unmasked)
                    loss_flex *= mask_flex
                    loss_flex = loss_flex.sum() / float(mask_flex.sum())  # averages the loss over the structure sequences
                    total_losses_test[2] += loss_flex.item()

                    total_losses_test[3] += loss_solvAcc.item() + loss_struct3.item()+ loss_struct8.item() + loss_flex.item()

                else:
                    loss_struct = self.crits[0](Y_struct_raw, Y_struct_true_unmasked)
                    loss_struct *= mask_struct
                    loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
                    total_losses_test[0] += loss_struct.item()

                    total_losses_test[3] += loss_struct.item()

                # In the case of multi-task learning on all targets either DSSP3 or DSSP8 classification is possible. This thesis only uses DSSP3 with multi-task but the code also
                # enables to use it with dssp8
                if(self.multi_mode=='multi4'):
                    if(self.dssp_mode==3):
                        Y_struct_pred_unmasked = torch.argmax(Y_struct_raw[0].data,dim=1)  # returns index of output predicted with highest probability
                        assert (torch.max(Y_struct_pred_unmasked) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked) >= 0)
                        Y_struct_pred = Y_struct_pred_unmasked[mask_struct[0] != 0]  # applies weighted mask
                        Y_struct_true = Y_struct_true_unmasked[0][mask_struct[0] != 0]  # applies weighted mask

                        #In the last epoch the performance is averaged over each sample in the test set to get a distribution of the performance measures and the standard error
                        if (final_flag):
                            for i in range(Y_struct_true_unmasked[0].size()[0]):
                                mask_struct_sample = mask_struct[0][i]
                                Y_struct_true_sample = Y_struct_true_unmasked[0][i]
                                Y_struct_pred_sample = Y_struct_pred_unmasked[i]
                                pred_dict.update({i: [Y_struct_pred_sample, len_wo_padding[i],mask_struct_sample]})
                                Y_struct_true_sample = Y_struct_true_sample[mask_struct_sample != 0]  # applies weighted mask
                                Y_struct_pred_sample = Y_struct_pred_sample[mask_struct_sample != 0]  # applies weighted mask

                                #If the mapping from DSSP8 to DSSP3 is used then mapping of predictions before evaluation
                                if (self.dssp_mode == 8 and self.map == 'True'):
                                    Y_struct_pred_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_pred_sample]
                                    Y_struct_true_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_true_sample]
                                    confusion_matrix_per_sample = np.zeros((3, 3), dtype=np.int)
                                    np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1)  # confusion matrix of one sample
                                    confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                                else:
                                    confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                                    np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1)  # confusion matrix of one sample
                                    confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                                confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                                np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample),1)  # confusion matrix of one sample
                                confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                                predarray_sa = np.array(Y_solvAcc_raw[i][0].cpu().numpy())
                                truearray_sa = np.array(Y_solvAcc_true_unmasked[i][0].cpu().numpy())
                                maskarray_sa = np.array(mask_solvAcc[i][0])
                                predarray_sa = predarray_sa[maskarray_sa != 0]
                                truearray_sa = truearray_sa[maskarray_sa != 0]

                                predarray_fl = np.array(Y_flex_raw[i][0].cpu().numpy())
                                truearray_fl = np.array(Y_flex_true_unmasked[i][0].cpu().numpy())
                                maskarray_fl = np.array(mask_flex[i][0])
                                predarray_fl = predarray_fl[maskarray_fl != 0]
                                truearray_fl = truearray_fl[maskarray_fl != 0]

                                rsquared_flex = r2_score(truearray_fl, predarray_fl)
                                rsquared_solvAcc = r2_score(truearray_sa, predarray_sa)

                                R2_list.append([rsquared_solvAcc, rsquared_flex])
                    #when dssp_mode==8
                    else:
                        Y_struct_pred_unmasked = torch.argmax(Y_struct_raw[1].data,dim=1)  # returns index of output predicted with highest probability
                        assert (torch.max(Y_struct_pred_unmasked) <= self.dssp_mode and torch.min( Y_struct_pred_unmasked) >= 0)
                        Y_struct_pred = Y_struct_pred_unmasked[mask_struct[1] != 0]  # applies weighted mask
                        Y_struct_true = Y_struct_true_unmasked[1][mask_struct[1] != 0]  # applies weighted mask

                        # In the last epoch the performance is averaged over each sample in the test set to get a distribution of the performance measures and the standard error
                        if (final_flag):
                            for i in range(Y_struct_true_unmasked[1].size()[0]):
                                mask_struct_sample = mask_struct[1][i]
                                Y_struct_true_sample = Y_struct_true_unmasked[1][i]
                                Y_struct_pred_sample = Y_struct_pred_unmasked[i]
                                pred_dict.update({i: [Y_struct_pred_sample, len_wo_padding[i],mask_struct_sample]})
                                Y_struct_true_sample = Y_struct_true_sample[mask_struct_sample != 0]  # applies weighted mask
                                Y_struct_pred_sample = Y_struct_pred_sample[mask_struct_sample != 0]  # applies weighted mask

                                if (self.dssp_mode == 8 and self.map == 'True'):
                                    Y_struct_true_sample = Y_struct_true_sample.view(-1).long().cpu().numpy()
                                    Y_struct_pred_sample = Y_struct_pred_sample.view(-1).long().cpu().numpy()
                                    Y_struct_pred_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_pred_sample]
                                    Y_struct_true_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_true_sample]
                                    confusion_matrix_per_sample = np.zeros((3, 3), dtype=np.int)
                                    np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1)  # confusion matrix of one sample
                                    confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                                else:
                                    confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode),dtype=np.int)
                                    np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1)  # confusion matrix of one sample
                                    confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                                predarray_sa = np.array(Y_solvAcc_raw[i][0].cpu().numpy())
                                truearray_sa = np.array(Y_solvAcc_true_unmasked[i][0].cpu().numpy())
                                maskarray_sa = np.array(mask_solvAcc[i][0])
                                predarray_sa = predarray_sa[maskarray_sa != 0]
                                truearray_sa = truearray_sa[maskarray_sa != 0]

                                predarray_fl = np.array(Y_flex_raw[i][0].cpu().numpy())
                                truearray_fl = np.array(Y_flex_true_unmasked[i][0].cpu().numpy())
                                maskarray_fl = np.array(mask_flex[i][0])
                                predarray_fl = predarray_fl[maskarray_fl != 0]
                                truearray_fl = truearray_fl[maskarray_fl != 0]

                                rsquared_flex = r2_score(truearray_fl, predarray_fl)
                                rsquared_solvAcc = r2_score(truearray_sa, predarray_sa)

                                R2_list.append([rsquared_solvAcc, rsquared_flex])

                # In all other cases the model determines the DSSPtype already and the structure targets are not organized in a list
                else:
                    Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data, dim=1)  # returns index of output predicted with highest probability
                    assert(torch.max(Y_struct_pred_unmasked)<=self.dssp_mode and torch.min(Y_struct_pred_unmasked) >= 0)
                    Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0] # applies weighted mask
                    Y_struct_true = Y_struct_true_unmasked[mask_struct != 0] # applies weighted mask

                    # In the last epoch the performance is averaged over each sample in the test set to get a distribution of the performance measures and the standard error
                    if (final_flag):
                        for i in range(Y_struct_true_unmasked.size()[0]):
                            mask_struct_sample=mask_struct[i]
                            Y_struct_true_sample=Y_struct_true_unmasked[i]
                            Y_struct_pred_sample=Y_struct_pred_unmasked[i]
                            pred_dict.update({i: [Y_struct_pred_sample, len_wo_padding[i],mask_struct_sample]})
                            Y_struct_true_sample=Y_struct_true_sample[mask_struct_sample!=0] # applies weighted mask
                            Y_struct_pred_sample=Y_struct_pred_sample[mask_struct_sample!=0] # applies weighted mask

                            if(self.dssp_mode==8 and self.map=='True'):
                                Y_struct_true_sample = Y_struct_true_sample.view(-1).long().cpu().numpy()
                                Y_struct_pred_sample = Y_struct_pred_sample.view(-1).long().cpu().numpy()
                                Y_struct_pred_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_pred_sample]
                                Y_struct_true_sample = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_true_sample]
                                confusion_matrix_per_sample = np.zeros((3,3),dtype=np.int)
                                np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1)  # confusion matrix of one sample
                                confmats_list.append(confusion_matrix_per_sample)  # collect them in list

                            else:
                                confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                                np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample),1)  # confusion matrix of one sample
                                confmats_list.append(confusion_matrix_per_sample) # collect them in list

                            predarray_sa = np.array(Y_solvAcc_raw[i][0].cpu().numpy())
                            truearray_sa = np.array(Y_solvAcc_true_unmasked[i][0].cpu().numpy())
                            maskarray_sa = np.array(mask_solvAcc[i][0])
                            predarray_sa = predarray_sa[maskarray_sa != 0]
                            truearray_sa = truearray_sa[maskarray_sa != 0]

                            predarray_fl = np.array(Y_flex_raw[i][0].cpu().numpy())
                            truearray_fl = np.array(Y_flex_true_unmasked[i][0].cpu().numpy())
                            maskarray_fl = np.array(mask_flex[i][0])
                            predarray_fl = predarray_fl[maskarray_fl != 0]
                            truearray_fl = truearray_fl[maskarray_fl != 0]

                            rsquared_flex = r2_score(truearray_fl, predarray_fl)
                            rsquared_solvAcc = r2_score(truearray_sa, predarray_sa)

                            R2_list.append([rsquared_solvAcc, rsquared_flex])

                #Collect true and predicted targets and generate confusion matrix
                Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
                Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

                Y_struct_true_accumulate_test = np.append(Y_struct_true_accumulate_test, Y_struct_true)
                Y_struct_pred_accumulate_test = np.append(Y_struct_pred_accumulate_test, Y_struct_pred)

                if (self.dssp_mode == 8 and self.map == 'True'):
                    Y_struct_pred = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_pred]
                    Y_struct_true = [DSSP8_TO_DSSP3_MAPPING[dssp8_state] for dssp8_state in Y_struct_true]

                np.add.at(confusion_matrix, (Y_struct_true, Y_struct_pred), 1) # confusion matrix for test step


            avg_losses_test[0] = round((total_losses_test[0] / len(test_loader)),3)
            avg_losses_test[1] = round((total_losses_test[1] / len(test_loader)),3)
            avg_losses_test[2] = round((total_losses_test[2] / len(test_loader)),3)
            avg_losses_test[3] = round((total_losses_test[3] / len(test_loader)),3)  # avg loss over total residues

            # plot solvAcc and Flex for one completely RANDOM sample of test set for checking if everything works correctly
            if (final_flag and self.multi_mode!='struct'):
                rnd=randint(0,Y_solvAcc_true_unmasked.shape[0]-1)
                print('-----------RANDOM SELECTED SAMPLE:', rnd,'--------------------------------')

                predarray_sa=np.array(Y_solvAcc_raw[rnd][0].cpu().numpy())
                truearray_sa = np.array(Y_solvAcc_true_unmasked[rnd][0].cpu().numpy())
                maskarray_sa=np.array(mask_solvAcc[rnd][0])
                predarray_sa=predarray_sa[maskarray_sa!=0]
                truearray_sa = truearray_sa[maskarray_sa != 0]

                predarray_fl = np.array(Y_flex_raw[rnd][0].cpu().numpy())
                truearray_fl = np.array(Y_flex_true_unmasked[rnd][0].cpu().numpy())
                maskarray_fl = np.array(mask_flex[rnd][0])
                predarray_fl = predarray_fl[maskarray_fl != 0]
                truearray_fl = truearray_fl[maskarray_fl != 0]

                utilities.writeSolvAccFlex(log_path, predarray_sa, truearray_sa, predarray_fl, truearray_fl)

            # Log the list of confusion matrices per sample for later analysis
            if(final_flag):
                if(self.dssp_mode==8 and self.map=='False'):
                    utilities.write_collected_confmats_8(log_path, confmats_list)
                else:
                    utilities.write_collected_confmats(log_path, confmats_list)
                utilities.write_predictedData(log_path, pred_dict)
                if(self.multi_mode!='struct'):
                    utilities.write_r2(log_path, R2_list)

        return avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train,avg_losses_test, Y_struct_true_accumulate_test, Y_struct_pred_accumulate_test, confusion_matrix


    # This function reads all data from the files and updates given dictionaries with it
    def getData(self, INPUT_PATH, TARGETS_PATH, protein, inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, input_type, class_type):
        if(input_type=='protvec+allpssm'):
            seq_scoring=np.load(INPUT_PATH + '/' + protein + '/protvec+scoringmatrix.npy')
            seq_info=np.load(INPUT_PATH + '/' + protein + '/protvec+information.npy')
            seq_info=seq_info[:,-1:]
            seq_weights=np.load(INPUT_PATH + '/' + protein + '/protvec+gapweights.npy')
            seq_weights = seq_weights[:, -1:]
            seq=np.concatenate((seq_scoring, seq_info, seq_weights), axis=1)
        elif(input_type=='pssm'):
            seq_scoring = np.load(INPUT_PATH + '/' + protein + '/protvec+scoringmatrix.npy')
            seq=seq_scoring[:,100:]
        else:
            seq = np.load(INPUT_PATH + '/' + protein + '/' + input_type + '.npy')

        tmp = {protein: seq}
        inputs.update(tmp)
        struct3 = np.load(INPUT_PATH + '/' + protein + '/structures_3.npy')
        struct8 = np.load(INPUT_PATH + '/' + protein + '/structures_8.npy')
        tmp = {protein: struct3}
        targets_structure3.update(tmp)
        tmp = {protein: struct8}
        targets_structure8.update(tmp)

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
        mask_solvAcc = np.ones(len(struct3))
        nans = np.argwhere(np.isnan(solvAcc.copy()))
        mask_solvAcc[nans] = 0
        tmp = {protein: mask_solvAcc}
        masks_solvAcc.update(tmp)
        solvAcc = np.nan_to_num(solvAcc)
        assert (np.min(solvAcc) >= 0 and np.max(solvAcc) <= 1)
        tmp = {protein: solvAcc.copy()}
        targets_solvAcc.update(tmp)
        del solvAcc

        mask_struct3 = np.load(INPUT_PATH + '/' + protein + '/mask_3.npy')
        mask_struct8 = np.load(INPUT_PATH + '/' + protein + '/mask_8.npy')
        tmp = {protein: mask_struct3}
        masks_struct3.update(tmp)
        tmp = {protein: mask_struct8}
        masks_struct8.update(tmp)