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

import DataLoader


class Pipeline():
    def __init__(self, dssp_mode, device, model, opt, crits, lstm_flag, multi_mode):
        self.dssp_mode=dssp_mode
        self.device=device
        self.model=model
        self.opt=opt
        self.crits=crits
        self.lstm_flag=lstm_flag
        self.multi_mode=multi_mode


    #
    # Training step
    #
    def trainNet(self, train_loader):

        self.model.train()

        for i, (X, Y_struct3_true, Y_struct8_true, Y_solvAcc_true, Y_flex_true, mask_struct3, mask_struct8, mask_solvAcc, mask_flex) in enumerate(train_loader): # iterate over batches

            X = X.to(self.device)
            if(self.multi_mode=='multi4'):
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

            if (self.lstm_flag):
                self.model.zero_grad()
                self.model.hidden = self.model.init_hidden(X.size()[0])

            self.opt.zero_grad()
            Y_struct_raw, Y_solvAcc_raw, Y_flex_raw = self.model(X)

            if(self.multi_mode=='multi2'):
                loss_struct = self.crits[0](Y_struct_raw, Y_struct_true)
                loss_struct *= mask_struct
                loss_struct = loss_struct.sum() / float(mask_struct.sum())  # averages the loss over the structure sequences

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(mask_solvAcc.sum())  # averages the loss over the structure sequences

                loss = loss_struct + loss_solvAcc

            elif(self.multi_mode=='multi3'):
                loss_struct = self.crits[0](Y_struct_raw, Y_struct_true)
                loss_struct *= mask_struct
                loss_struct = loss_struct.sum() / float(
                    mask_struct.sum())  # averages the loss over the structure sequences

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(
                    mask_solvAcc.sum())  # averages the loss over the structure sequences

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
                loss_struct3 = loss_struct3.sum() / float(mask_struct[0].sum())  # averages the loss over the structure sequences

                loss_struct8 = self.crits[0](Y_struct_raw[1], Y_struct_true[1])
                loss_struct8 *= mask_struct[1]
                loss_struct8 = loss_struct8.sum() / float(mask_struct[1].sum())  # averages the loss over the structure sequences

                loss_solvAcc = self.crits[1](Y_solvAcc_raw, Y_solvAcc_true)
                loss_solvAcc *= mask_solvAcc
                loss_solvAcc = loss_solvAcc.sum() / float(
                    mask_solvAcc.sum())  # averages the loss over the structure sequences

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
                loss_struct = loss_struct.sum() / float(mask_struct.sum())  # averages the loss over the structure sequences

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
            total_losses_test=[0,0,0,0]
            avg_losses_test=[0,0,0,0] #[struct,solvAcc,flex,sum]
            pred_dict=dict()

            Y_struct_true_accumulate_test = np.zeros(0, dtype=np.int) # collects true targets
            Y_struct_pred_accumulate_test = np.zeros(0, dtype=np.int) # collects predicted targets
            confusion_matrix=np.zeros((self.dssp_mode,self.dssp_mode), dtype=np.int) # confusion matrix
            bootstrapped_conf_mat=[] # collects confusion matrices for each sample in last epoch

            def evaluateTrainLoss():
                with torch.no_grad():
                    total_losses_train=[0,0,0,0]
                    avg_losses_train=[0,0,0,0]

                    Y_struct_true_accumulate_train = np.zeros(0, dtype=np.int)  # collects true targets
                    Y_struct_pred_accumulate_train = np.zeros(0, dtype=np.int)  # collects predicted targets


                    for X_train, Y_struct3_train,Y_struct8_train, Y_solvAcc_train, Y_flex_train, mask_struct3_train,mask_struct8_train, mask_solvAcc_train, mask_flex_train in train_loader:  # iterate over batches

                        X_train = X_train.to(self.device)

                        if(self.multi_mode=='multi4'):
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

                        if (self.lstm_flag):
                            self.model.zero_grad()
                            self.model.hidden = self.model.init_hidden(X_train.size()[0])

                        Y_struct_raw_train, Y_solvAcc_raw_train, Y_flex_raw_train = self.model(X_train)

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

            avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train=evaluateTrainLoss()


            for X, Y_struct3_true,Y_struct8_true, Y_solvAcc_true, Y_flex_true, mask_struct3, mask_struct8, mask_solvAcc, mask_flex in test_loader: # iterate over batches
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

                if (self.lstm_flag):
                    self.model.zero_grad()
                    self.model.hidden = self.model.init_hidden(X.size()[0])

                Y_struct_raw,  Y_solvAcc_raw, Y_flex_raw = self.model(X)

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

                if(self.multi_mode=='multi4'):
                    if(self.dssp_mode==3):
                        Y_struct_pred_unmasked = torch.argmax(Y_struct_raw[0].data,dim=1)  # returns index of output predicted with highest probability
                        assert (torch.max(Y_struct_pred_unmasked) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked) >= 0)
                        Y_struct_pred = Y_struct_pred_unmasked[mask_struct[0] != 0]  # applies weighted mask
                        Y_struct_true = Y_struct_true_unmasked[0][mask_struct[0] != 0]  # applies weighted mask

                        if (final_flag):  # last epoch, bootstrapping only over the test samples --> len(bootstrapped_conf_mat)==len(validation_set)
                            print('BOOTSTRAPPING', Y_struct_true_unmasked[0].size())
                            for i in range(Y_struct_true_unmasked[0].size()[0]):
                                mask_struct_sample = mask_struct[0][i]
                                Y_struct_true_sample = Y_struct_true_unmasked[0][i]
                                Y_struct_pred_sample = Y_struct_pred_unmasked[i]
                                Y_struct_true_sample = Y_struct_true_sample[mask_struct_sample != 0]  # applies weighted mask
                                Y_struct_pred_sample = Y_struct_pred_sample[mask_struct_sample != 0]  # applies weighted mask
                                confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                                np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample),1)  # confusion matrix of one sample
                                bootstrapped_conf_mat.append(confusion_matrix_per_sample)  # collect them in list
                                pred_dict.update({i: Y_struct_pred_sample})


                    else:
                        Y_struct_pred_unmasked = torch.argmax(Y_struct_raw[1].data,dim=1)  # returns index of output predicted with highest probability
                        assert (torch.max(Y_struct_pred_unmasked) <= self.dssp_mode and torch.min( Y_struct_pred_unmasked) >= 0)
                        Y_struct_pred = Y_struct_pred_unmasked[mask_struct[1] != 0]  # applies weighted mask
                        Y_struct_true = Y_struct_true_unmasked[1][mask_struct[1] != 0]  # applies weighted mask

                        if (final_flag):  # last epoch, bootstrapping only over the test samples --> len(bootstrapped_conf_mat)==len(validation_set)
                            print('BOOTSTRAPPING', Y_struct_true_unmasked[1].size())
                            for i in range(Y_struct_true_unmasked[1].size()[0]):
                                mask_struct_sample = mask_struct[1][i]
                                Y_struct_true_sample = Y_struct_true_unmasked[1][i]
                                Y_struct_pred_sample = Y_struct_pred_unmasked[i]
                                Y_struct_true_sample = Y_struct_true_sample[
                                    mask_struct_sample != 0]  # applies weighted mask
                                Y_struct_pred_sample = Y_struct_pred_sample[
                                    mask_struct_sample != 0]  # applies weighted mask
                                confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                                np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample),
                                          1)  # confusion matrix of one sample
                                bootstrapped_conf_mat.append(confusion_matrix_per_sample)  # collect them in list
                                pred_dict.update({i: Y_struct_pred_sample})


                else:

                    Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data, dim=1)  # returns index of output predicted with highest probability
                    assert(torch.max(Y_struct_pred_unmasked)<=self.dssp_mode and torch.min(Y_struct_pred_unmasked) >= 0)
                    Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0] # applies weighted mask

                    Y_struct_true = Y_struct_true_unmasked[mask_struct != 0] # applies weighted mask

                    if (final_flag): # last epoch, bootstrapping only over the test samples --> len(bootstrapped_conf_mat)==len(validation_set)
                        print('BOOTSTRAPPING',Y_struct_true_unmasked.size())
                        for i in range(Y_struct_true_unmasked.size()[0]):
                            mask_struct_sample=mask_struct[i]
                            Y_struct_true_sample=Y_struct_true_unmasked[i]
                            Y_struct_pred_sample=Y_struct_pred_unmasked[i]
                            Y_struct_true_sample=Y_struct_true_sample[mask_struct_sample!=0] # applies weighted mask
                            Y_struct_pred_sample=Y_struct_pred_sample[mask_struct_sample!=0] # applies weighted mask
                            confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                            np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample), 1) # confusion matrix of one sample
                            bootstrapped_conf_mat.append(confusion_matrix_per_sample) # collect them in list
                            pred_dict.update({i: Y_struct_pred_sample})

                Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
                Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

                np.add.at(confusion_matrix, (Y_struct_true, Y_struct_pred), 1) # confusion matrix for test step
                Y_struct_true_accumulate_test = np.append(Y_struct_true_accumulate_test, Y_struct_true)
                Y_struct_pred_accumulate_test = np.append(Y_struct_pred_accumulate_test, Y_struct_pred)

            avg_losses_test[0] = round((total_losses_test[0] / len(test_loader)),3)
            avg_losses_test[1] = round((total_losses_test[1] / len(test_loader)),3)
            avg_losses_test[2] = round((total_losses_test[2] / len(test_loader)),3)
            avg_losses_test[3] = round((total_losses_test[3] / len(test_loader)),3)  # avg loss over total residues

            if (final_flag and self.multi_mode!='struct'): #plot solvAcc and Flex for one random sample for checking if everything works correctly
                predarray_sa=np.array(Y_solvAcc_raw[4][0].cpu().numpy())
                truearray_sa = np.array(Y_solvAcc_true_unmasked[4][0].cpu().numpy())
                maskarray_sa=np.array(mask_solvAcc[4][0])
                predarray_sa=predarray_sa[maskarray_sa!=0]
                truearray_sa = truearray_sa[maskarray_sa != 0]

                predarray_fl = np.array(Y_flex_raw[4][0].cpu().numpy())
                truearray_fl = np.array(Y_flex_true_unmasked[4][0].cpu().numpy())
                maskarray_fl = np.array(mask_flex[4][0])
                predarray_fl = predarray_fl[maskarray_fl != 0]
                truearray_fl = truearray_fl[maskarray_fl != 0]

                rsquared_flex = r2_score(truearray_fl, predarray_fl)
                rsquared_solvAcc = r2_score(truearray_sa, predarray_sa)

                print('R2 ERROR: ', rsquared_flex, rsquared_solvAcc)

                utilities.writeSolvAccFlex(log_path, predarray_sa, truearray_sa, predarray_fl, truearray_fl)

            if(final_flag): # plotting of bootstrapping in last epoch
                print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
                #utilities.plot_bootstrapping(log_path, bootstrapped_conf_mat)
                utilities.write_bootstrapping(log_path, bootstrapped_conf_mat)
                utilities.write_predictedData(log_path, pred_dict)

        return avg_losses_train, Y_struct_true_all_train, Y_struct_pred_all_train,avg_losses_test, Y_struct_true_accumulate_test, Y_struct_pred_accumulate_test, confusion_matrix





    def getData(self, INPUT_PATH, TARGETS_PATH, protein, inputs, targets_structure3, targets_structure8, targets_solvAcc, targets_flexibility, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, input_type, class_type):

        #if (os.path.isfile(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap')):

        if(input_type=='protvec+allpssm'):
            seq_scoring=np.load(INPUT_PATH + '/' + protein + '/protvec+scoringmatrix.npy')
            seq_info=np.load(INPUT_PATH + '/' + protein + '/protvec+information.npy')
            seq_info=seq_info[:,-1:]
            assert(seq_info.shape==(seq_scoring.shape[0],1))
            seq_weights=np.load(INPUT_PATH + '/' + protein + '/protvec+gapweights.npy')
            seq_weights = seq_weights[:, -1:]
            assert (seq_weights.shape == (seq_scoring.shape[0], 1))
            seq=np.concatenate((seq_scoring, seq_info, seq_weights), axis=1)
            assert(seq.shape== (seq_scoring.shape[0], 122))
        elif(input_type=='pssm'):
            seq_scoring = np.load(INPUT_PATH + '/' + protein + '/protvec+scoringmatrix.npy')
            seq=seq_scoring[:,100:]
            assert(seq.shape==(seq_scoring.shape[0], 20))
        else:
            seq = np.load(INPUT_PATH + '/' + protein + '/' + input_type + '.npy')
            #print(seq)

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


















