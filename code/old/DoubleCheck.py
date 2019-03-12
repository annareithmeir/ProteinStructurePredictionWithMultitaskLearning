import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import  utilities
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class DoubleCheck():

    def __init__(self, test_loader, dssp_mode, device, model, opt, crits, lstm_flag, multi_flag):
        self.dssp_mode=dssp_mode
        self.device=device
        self.model=model
        self.opt=opt
        self.crits=crits
        self.lstm_flag=lstm_flag
        self.test_loader=test_loader
        self.multi_flag=multi_flag

    def evaluateModel(self):
        self.model.eval()

        with torch.no_grad():
            total_losses_test = [0, 0, 0, 0]
            avg_losses_test = [0, 0, 0, 0]  # [struct,solvAcc,flex,sum]

            Y_struct_true_all = np.zeros(0, dtype=np.int)  # collects true targets
            Y_struct_pred_all = np.zeros(0, dtype=np.int)  # collects predicted targets
            confusion_matrix = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)  # confusion matrix
            bootstrapped_conf_mat = []  # collects confusion matrices for each sample in last epoch

            for X, Y_struct_true, Y_solvAcc_true, Y_flex_true, mask_struct, mask_solvAcc, mask_flex in self.test_loader:  # iterate over batches
                X = X.to(self.device)

                Y_struct_true_unmasked = Y_struct_true.to(self.device)
                Y_solvAcc_true_unmasked = Y_solvAcc_true.to(self.device)
                Y_flex_true_unmasked = Y_flex_true.to(self.device)
                mask_struct = mask_struct.to(self.device)
                mask_solvAcc = mask_solvAcc.to(self.device)
                mask_flex = mask_flex.to(self.device)

                if (self.lstm_flag):
                    self.model.zero_grad()
                    self.model.hidden = self.model.init_hidden(X.size()[0])

                Y_struct_raw, Y_solvAcc_raw, Y_flex_raw = self.model(X)

                if (self.multi_flag):
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
                    loss_flex = loss_flex.sum() / float(mask_flex.sum())  # averages the loss over the structure sequences
                    total_losses_test[2] += loss_flex.item()

                    total_losses_test[3] += loss_solvAcc.item() + loss_struct.item() + loss_flex.item()
                else:
                    loss_struct = self.crits[0](Y_struct_raw, Y_struct_true_unmasked)
                    loss_struct *= mask_struct
                    loss_struct = (loss_struct.sum()) / float(mask_struct.sum())
                    total_losses_test[0] += loss_struct.item()

                    total_losses_test[3] += loss_struct.item()

                Y_struct_pred_unmasked = torch.argmax(Y_struct_raw.data,
                                                      dim=1)  # returns index of output predicted with highest probability
                assert (torch.max(Y_struct_pred_unmasked) <= self.dssp_mode and torch.min(Y_struct_pred_unmasked) >= 0)
                Y_struct_pred = Y_struct_pred_unmasked[mask_struct != 0]  # applies weighted mask

                Y_struct_true = Y_struct_true_unmasked[mask_struct != 0]  # applies weighted mask

                '''
                if (
                final_flag):  # last epoch, bootstrapping only over the test samples --> len(bootstrapped_conf_mat)==len(validation_set)
                    print('BOOTSTRAPPING', Y_struct_true_unmasked.size())
                    for i in range(Y_struct_true_unmasked.size()[0]):
                        mask_struct_sample = mask_struct[i]
                        Y_struct_true_sample = Y_struct_true_unmasked[i]
                        Y_struct_pred_sample = Y_struct_pred_unmasked[i]
                        Y_struct_true_sample = Y_struct_true_sample[mask_struct_sample != 0]  # applies weighted mask
                        Y_struct_pred_sample = Y_struct_pred_sample[mask_struct_sample != 0]  # applies weighted mask
                        confusion_matrix_per_sample = np.zeros((self.dssp_mode, self.dssp_mode), dtype=np.int)
                        np.add.at(confusion_matrix_per_sample, (Y_struct_true_sample, Y_struct_pred_sample),
                                  1)  # confusion matrix of one sample
                        bootstrapped_conf_mat.append(confusion_matrix_per_sample)  # collect them in list
                '''

                Y_struct_true = Y_struct_true.view(-1).long().cpu().numpy()
                Y_struct_pred = Y_struct_pred.view(-1).long().cpu().numpy()

                np.add.at(confusion_matrix, (Y_struct_true, Y_struct_pred), 1)  # confusion matrix for test step
                print(confusion_matrix)
                Y_struct_true_all_test = np.append(Y_struct_true_all, Y_struct_true)
                Y_struct_pred_all_test = np.append(Y_struct_pred_all, Y_struct_pred)

            avg_losses_test[0] = round((total_losses_test[0] / len(self.test_loader)), 3)
            avg_losses_test[1] = round((total_losses_test[1] / len(self.test_loader)), 3)
            avg_losses_test[2] = round((total_losses_test[2] / len(self.test_loader)), 3)
            avg_losses_test[3] = round((total_losses_test[3] / len(self.test_loader)), 3)  # avg loss over total residues

            if (self.multi_flag):  # plot solvAcc and Flex for one random sample for checking if everything works correctly
                predarray_sa = np.array(Y_solvAcc_raw[0][0][:100].cpu().numpy())
                truearray_sa = np.array(Y_solvAcc_true_unmasked[0][0][:100].cpu().numpy())
                maskarray_sa = np.array(mask_solvAcc[0][0][:100])
                predarray_sa = predarray_sa[maskarray_sa != 0]
                truearray_sa = truearray_sa[maskarray_sa != 0]

                predarray_fl = np.array(Y_flex_raw[0][0][:100].cpu().numpy())
                truearray_fl = np.array(Y_flex_true_unmasked[0][0][:100].cpu().numpy())
                maskarray_fl = np.array(mask_flex[0][0][:100])
                predarray_fl = predarray_fl[maskarray_fl != 0]
                truearray_fl = truearray_fl[maskarray_fl != 0]

                # print(truearray_sa.shape,predarray_sa.shape,'--->')
                import matplotlib.pyplot as plt
                plt.clf()
                # fig, (axes) = plt.subplots(2, 1, figsize=(28, 14), sharex=True, sharey=True)
                plt.subplot(2, 1, 1)
                # print('#######', predarray.size)
                plt.plot(np.arange(len(predarray_sa)), predarray_sa, color='red')
                plt.plot(np.arange(len(truearray_sa)), truearray_sa, color='green')
                plt.xlabel('SolvAcc')
                plt.title('SolvAcc and Flex predictions')

                plt.subplot(2, 1, 2)
                plt.plot(np.arange(len(predarray_fl)), predarray_fl, color='red')
                plt.plot(np.arange(len(truearray_fl)), truearray_fl, color='green')
                plt.xlabel('Flex')
                axes = plt.gca()
                axes.set_ylim([-1, 2])
                plt.savefig('CheckSolvAccFlexSampleLinReg_2.pdf')

            '''
            if (final_flag):  # plotting of bootstrapping in last epoch
                print('# BOOTSTRAPPING SAMPLES: ', len(bootstrapped_conf_mat))
                utilities.plot_bootstrapping(log_path, bootstrapped_conf_mat)
            '''

        return avg_losses_test, Y_struct_true_all_test, Y_struct_pred_all_test, confusion_matrix


+print('-----EVAL-----')

eval=DoubleCheck.DoubleCheck(data_loaders['Eval'],DSSP_MODE, device, model, opt, criterions, False, True)

avgloss, truelabels, predlabels, confmat=eval.evaluateModel()

print(avgloss)
print(round(accuracy_score( truelabels, predlabels ),2))


