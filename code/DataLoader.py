import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

#
# This file handles data loading
#


# These two helper methods calculate the proportions of each class --> to ensure that all classes are present in splitted sets
def countStructures3(train, test, val):
    def countStructs(dict):
        c = 0
        h = 0
        e = 0

        for sample in dict.keys():
            sequence = dict[sample][1]
            mask_struct=dict[sample][4]
            for res in range(len(sequence)):
                if (sequence[res] == 0 and mask_struct[res]!=0):
                    c += 1
                elif (sequence[res] == 1 and mask_struct[res]!=0):
                    h += 1
                elif (sequence[res] == 2 and mask_struct[res]!=0):
                    e += 1
                elif (mask_struct[res]==0):
                    pass
                else:
                    raise ValueError('Unknown structure', sequence[res])

        return c, h, e

    # check if classes occur around same times in train and test set
    ctrain, htrain, etrain = countStructs(train)
    trainsum = ctrain + htrain + etrain
    ctest, htest, etest = countStructs(test)
    testsum = ctest + htest + etest
    cval, hval, eval = countStructs(val)
    valsum = cval + hval + eval
    print('training structure ratio %:', round(ctrain / float(trainsum)*100,2), round(htrain / float(trainsum)*100,2), round(etrain / float(trainsum)*100,2))
    print('testing structure ratio %:', round(ctest / float(testsum)*100,2), round(htest / float(testsum)*100,2), round(etest / float(testsum)*100,2))
    print('validation structure ratio %:', round(cval / float(valsum)*100,2), round(hval / float(valsum)*100,2), round(eval / float(valsum)*100,2))
def countStructures8(train, test, val):
    def countStructs(dict):
        h = 0
        e = 0
        i=0
        s=0
        t=0
        g=0
        b=0
        none=0

        for sample in dict.keys():
            sequence = dict[sample][1]
            mask_struct=dict[sample][4]
            for res in range(len(sequence)):
                if (sequence[res] == 0 and mask_struct[res]!=0):
                    h += 1
                elif (sequence[res] == 1 and mask_struct[res]!=0):
                    e += 1
                elif (sequence[res] == 2 and mask_struct[res]!=0):
                    i += 1
                elif (sequence[res] == 3 and mask_struct[res]!=0):
                    s += 1
                elif (sequence[res] == 4 and mask_struct[res]!=0):
                    t += 1
                elif (sequence[res] == 5 and mask_struct[res]!=0):
                    g += 1
                elif (sequence[res] == 6 and mask_struct[res]!=0):
                    b += 1
                elif (sequence[res] == 7 and mask_struct[res]!=0):
                    none += 1
                elif (mask_struct[res]==0):
                    pass
                else:
                    raise ValueError('Unknown structure', sequence[res])

        return h, e, i, s, t, g, b, none

    # check if classes occur around same times in train and test set
    htrain, etrain, itrain, strain, ttrain, gtrain, btrain, nonetrain = countStructs(train)
    trainsum = htrain+ etrain+ itrain+ strain+ ttrain+ gtrain+ btrain+ nonetrain
    htest, etest, itest, stest, ttest, gtest, btest, nonetest = countStructs(test)
    testsum = htest+ etest+ itest+ stest+ ttest+ gtest+ btest+ nonetest
    hval, eval, ival, sval, tval, gval, bval, noneval = countStructs(val)
    valsum = hval+ eval+ ival+ sval+ tval+ gval+ bval+ noneval
    print('training structure ratio %:', round(htrain / float(trainsum)*100,2), round(etrain / float(trainsum)*100,2),round(itrain / float(trainsum)*100,2),round(strain / float(trainsum)*100,2),round(ttrain / float(trainsum)*100,2),round(gtrain / float(trainsum)*100,2),round(btrain / float(trainsum)*100,2),round(nonetrain / float(trainsum)*100,2))
    print('testing structure ratio %:',  round(htest / float(testsum)*100,2), round(etest / float(testsum)*100,2), round(itest / float(testsum)*100,2), round(stest / float(testsum)*100,2), round(ttest / float(testsum)*100,2), round(gtest / float(testsum)*100,2), round(btest / float(testsum)*100,2), round(nonetest / float(testsum)*100,2))
    print('validation structure ratio %:',  round(hval / float(valsum)*100,2), round(eval / float(valsum)*100,2), round(ival / float(valsum)*100,2), round(sval / float(valsum)*100,2), round(tval / float(valsum)*100,2), round(gval / float(valsum)*100,2), round(bval / float(valsum)*100,2), round(noneval / float(valsum)*100,2))


# Splits into train, validation and test set randomly
class Data_splitter():
    def __init__(self, inputs, targets_struct3, targets_struct8, targets_solvAcc, targets_flex, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, dssp_mode, validation_ratio=0.2,test_ratio=0.2):  # 20% test --> rest: 80% train, 20% validation
        self.inputs = inputs
        self.targets_struct3 = targets_struct3
        self.targets_struct8=targets_struct8
        self.targets_solvAcc= targets_solvAcc
        self.targets_flex=targets_flex
        self.masks_struct3=masks_struct3
        self.masks_struct8=masks_struct8
        self.masks_solvAcc=masks_solvAcc
        self.masks_flex=masks_flex
        self.test_ratio = test_ratio
        self.validation_ratio=validation_ratio
        self.max_len = 0
        self.dssp_mode=dssp_mode

    #To be used if data is not yet splitted
    def split_data(self):
        train = dict()
        val=dict()
        test = dict()
        cnt_test = 0
        cnt_train = 0
        cnt_val=0
        np.random.seed(987)  # to reproduce splits
        proteins=self.inputs.keys()
        for i in proteins:
            rnd = np.random.rand()

            if (len(self.targets_struct3[i]) > self.max_len):
                self.max_len = len(self.targets_struct3[i])
            if (rnd > (1 - self.test_ratio)):
                test[cnt_test] = [self.inputs[i], self.targets_struct3[i], self.targets_struct8[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct3[i], self.masks_struct8[i],self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_test += 1
            elif(rnd<= (1-self.test_ratio) and rnd> (1-self.test_ratio)*self.validation_ratio):
                train[cnt_train] = [self.inputs[i], self.targets_struct3[i], self.targets_struct8[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct3[i], self.masks_struct8[i], self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_train += 1
            else:
                val[cnt_val] = [self.inputs[i], self.targets_struct3[i], self.targets_struct8[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct3[i], self.masks_struct8[i], self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_val += 1

        print('Size of training set: {}'.format(len(train)))
        print('Size of test set: {}'.format(len(test)))
        print('Size of validation set: {}'.format(len(val)))
        if(self.dssp_mode==8):
            countStructures8(train, test, val)
        else:
            countStructures3(train, test, val)
        return train, val, test

    #To be used if data is already splitted
    def make_data(self):
        with open('dataset_overview.txt', 'w') as file:
            data = dict()
            cnt = 0
            proteins=self.inputs.keys()
            for i in proteins:
                file.write(str(i+'  -  '+str(cnt)))
                file.write('\n')
                if (len(self.targets_struct3[i]) > self.max_len):
                    self.max_len = len(self.targets_struct3[i])
                data[cnt] = [self.inputs[i], self.targets_struct3[i], self.targets_struct8[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct3[i], self.masks_struct8[i],self.masks_solvAcc[i], self.masks_flex[i]]
                cnt += 1

            return data

    #Returns max length in data set
    def get_max_length(self):
        return self.max_len


#
# Custom dataset class, returns all inputs, targets and masks for one sample.
# Each sample is padded with zeros
#

class Dataset(data.Dataset):

    def __init__(self, samples, weights, dssp_mode, max_len=None):
        self.max_len = max_len
        self.inputs, self.targets_struct3, self.targets_struct8, self.targets_solvAcc, self.targets_flex, self.masks_struct3, self.masks_struct8, self.masks_solvAcc, self.masks_flex = zip(*[[inputs, targets_struct3, targets_struct8,  targets_solvAcc, targets_flex, masks_struct3, masks_struct8,  masks_solvAcc, masks_flex]
                                          for _, [inputs, targets_struct3, targets_struct8,  targets_solvAcc, targets_flex, masks_struct3, masks_struct8,  masks_solvAcc, masks_flex] in samples.items()])
        self.data_len = len(self.inputs)
        self.to_tensor = transforms.ToTensor()
        self.weights=weights
        self.dssp_mode=dssp_mode

    def __getitem__(self, idx):
        X = self.inputs[idx]
        Y_struct3 = self.targets_struct3[idx]
        Y_struct8 = self.targets_struct8[idx]
        Y_solvAcc = self.targets_solvAcc[idx]
        Y_flex= self.targets_flex[idx]
        mask_struct3=self.masks_struct3[idx]
        mask_struct8 = self.masks_struct8[idx]
        #mask_struct = self.masks_struct[idx]
        mask_solvAcc=self.masks_solvAcc[idx]
        mask_flex=self.masks_flex[idx]
        length_wo_padding=len(mask_struct3)

        assert(len(Y_struct3)==len(Y_struct8)==len(mask_struct3)==len(Y_solvAcc)==len(mask_solvAcc))

        #Padding
        if self.max_len:
            n_missing = self.max_len - len(Y_struct3)
            Y_struct3 = np.pad(Y_struct3, pad_width=(0, n_missing), mode='constant', constant_values=0)
            Y_struct8 = np.pad(Y_struct8, pad_width=(0, n_missing), mode='constant', constant_values=0)
            Y_solvAcc = np.pad(Y_solvAcc, pad_width=(0, n_missing), mode='constant', constant_values=0)
            Y_flex = np.pad(Y_flex, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_struct3 = np.pad(mask_struct3, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_struct8 = np.pad(mask_struct8, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_solvAcc = np.pad(mask_solvAcc, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_flex = np.pad(mask_flex, pad_width=(0, n_missing), mode='constant', constant_values=0)
            X = np.pad(X, pad_width=((0, n_missing), (0, 0)), mode='constant', constant_values=0.)

        X = X.T
        X = np.expand_dims(X, axis=1)
        Y_struct3 = np.expand_dims(Y_struct3, axis=1)
        Y_struct8 = np.expand_dims(Y_struct8, axis=1)
        Y_solvAcc = np.expand_dims(Y_solvAcc, axis=1)
        Y_flex = np.expand_dims(Y_flex, axis=1)
        mask_struct3 = np.expand_dims(mask_struct3, axis=0)
        mask_struct8 = np.expand_dims(mask_struct8, axis=0)
        mask_solvAcc = np.expand_dims(mask_solvAcc, axis=0)
        mask_flex = np.expand_dims(mask_flex, axis=0)
        Y_struct3 = Y_struct3.T
        Y_struct8=Y_struct8.T
        Y_solvAcc=Y_solvAcc.T
        Y_flex = Y_flex.T

        #Data to PyTorch tensors
        input_tensor = torch.from_numpy(X).type(dtype=torch.float)
        struct3_tensor = torch.from_numpy(Y_struct3)
        struct8_tensor = torch.from_numpy(Y_struct8)
        solvAcc_tensor=torch.from_numpy(Y_solvAcc)
        flex_tensor = torch.from_numpy(Y_flex)
        mask_struct3_tensor = torch.from_numpy(mask_struct3).type(dtype=torch.float)
        mask_struct8_tensor = torch.from_numpy(mask_struct8).type(dtype=torch.float)
        #mask_struct_tensor = torch.from_numpy(mask_struct).type(dtype=torch.float)
        mask_solvAcc_tensor = torch.from_numpy(mask_solvAcc).type(dtype=torch.float)
        mask_flex_tensor = torch.from_numpy(mask_flex).type(dtype=torch.float)

        #Apply class balancing for DSSP3
        mask_struct3_tensor = torch.Tensor(np.array([self.weights[0][i] for i in struct3_tensor])) * mask_struct3_tensor

        return (input_tensor, struct3_tensor, struct8_tensor,  solvAcc_tensor, flex_tensor, mask_struct3_tensor, mask_struct8_tensor,  mask_solvAcc_tensor, mask_flex_tensor, length_wo_padding)

    #Return size of data set
    def __len__(self):
        return self.data_len


#To be used when data not splitted yet. Makes three Dataset() out of the whole dataset.
def train_val_test_split(inputs, targets_struct3, targets_struct8, targets_solcAcc, targets_flex, masks_struct3, masks_struct8, masks_solvAcc, masks_flex,weights, dssp_mode):
    d = Data_splitter(inputs, targets_struct3, targets_struct8, targets_solcAcc, targets_flex, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, dssp_mode)
    train, val, test = d.split_data()
    max_len = d.get_max_length()
    train_set = Dataset(train, weights,dssp_mode, max_len)
    val_set = Dataset(val,weights, dssp_mode, max_len)
    test_set = Dataset(test, weights, dssp_mode,  max_len)
    return train_set, val_set, test_set

#To be used when data set already splitted and either train/test/val set given. Makes Dataset() out of dataset.
def make_dataset(inputs, targets_struct3, targets_struct8, targets_solcAcc, targets_flex, masks_struct3, masks_struct8, masks_solvAcc, masks_flex,weights, dssp_mode):
    d = Data_splitter(inputs, targets_struct3, targets_struct8, targets_solcAcc, targets_flex, masks_struct3, masks_struct8, masks_solvAcc, masks_flex, dssp_mode)
    data = d.make_data()
    max_len = d.get_max_length()
    data_set = Dataset(data, weights,dssp_mode, max_len)
    return data_set


# Creates Dataloaders of specified batchsizes. Either Train + Validation or Train + Test.
def createDataLoaders(train_set, test_set, batch_size_train, batch_size_test):
    data_loaders = dict()
    data_loaders['Train'] = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=batch_size_train,
                                                        shuffle=True,
                                                        drop_last=False
                                                        )
    data_loaders['Test'] = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=batch_size_test,
                                                       shuffle=False,
                                                       drop_last=False
                                                       )
    return data_loaders