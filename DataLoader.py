import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import utilities_MICHAEL as utilities

#
# This method calculates the proportions of each class --> to ensure that all classes are present in splitted sets
#
def countStructures3(train, test, val):
    def countStructs(dict):
        c = 0
        h = 0
        e = 0
        for sample in dict.keys():
            sequence = dict[sample][1]
            mask_struct=dict[sample][2]
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


#
# Splits data into training and testing set
#
class Data_splitter():
    def __init__(self, inputs, targets_struct, targets_solvAcc, targets_flex, masks_struct, masks_solvAcc, masks_flex, validation_ratio=0.2,test_ratio=0.2):  # 20% test, rest: 80% train, 20% validation
        self.inputs = inputs
        self.targets_struct = targets_struct
        self.targets_solvAcc= targets_solvAcc
        self.targets_flex=targets_flex
        self.masks_struct=masks_struct
        self.masks_solvAcc=masks_solvAcc
        self.masks_flex=masks_flex
        self.test_ratio = test_ratio
        self.validation_ratio=validation_ratio
        self.max_len = 0

    def split_data(self):
        train = dict()
        val=dict()
        test = dict()
        cnt_test = 0
        cnt_train = 0
        cnt_val=0
        np.random.seed(42)  # to reproduce splits
        proteins=self.inputs.keys()
        for i in proteins:
            rnd = np.random.rand()

            if (len(self.targets_struct[i]) > self.max_len):
                self.max_len = len(self.targets_struct[i])
            if (rnd > (1 - self.test_ratio)):
                test[cnt_test] = [self.inputs[i], self.targets_struct[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct[i], self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_test += 1
            elif(rnd< (1-self.test_ratio) and rnd> (1-self.test_ratio)*self.validation_ratio):
                train[cnt_train] = [self.inputs[i], self.targets_struct[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct[i], self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_train += 1
            else:
                val[cnt_val] = [self.inputs[i], self.targets_struct[i], self.targets_solvAcc[i], self.targets_flex[i], self.masks_struct[i], self.masks_solvAcc[i], self.masks_flex[i]]
                cnt_val += 1

        print('Size of training set: {}'.format(len(train)))
        print('Size of test set: {}'.format(len(test)))
        print('Size of validation set: {}'.format(len(val)))
        #countStructures3(train, test, val)

        return train, val, test

    def get_max_length(self):
        return self.max_len


#
# Custom dataset class, returns (ProtVec/1hot, target, weighted mask) for one sample.
# Each sample is padded with zeros and has length 968
#
class Dataset(data.Dataset):

    def __init__(self, samples, weights, max_len=None):
        self.max_len = max_len
        self.inputs, self.targets_struct, self.targets_solvAcc, self.targets_flex, self.masks_struct, self.masks_solvAcc, self.masks_flex = zip(*[[inputs, targets_struct, targets_solvAcc, targets_flex, masks_struct, masks_solvAcc, masks_flex]
                                          for _, [inputs, targets_struct, targets_solvAcc, targets_flex, masks_struct, masks_solvAcc, masks_flex] in samples.items()])
        self.data_len = len(self.inputs)
        self.to_tensor = transforms.ToTensor()
        self.weights=weights

    def __getitem__(self, idx):
        X = self.inputs[idx]
        Y_struct = self.targets_struct[idx]
        Y_solvAcc = self.targets_solvAcc[idx]
        Y_flex= self.targets_flex[idx]
        mask_struct=self.masks_struct[idx]
        mask_solvAcc=self.masks_solvAcc[idx]
        mask_flex=self.masks_flex[idx]

        if self.max_len:  # if a maximum length (longest seq. in the set) was provided
            n_missing = self.max_len - len(
                Y_struct)  # get the number of residues which have to be padded to achieve same size for all proteins in the set
            Y_struct = np.pad(Y_struct, pad_width=(0, n_missing), mode='constant', constant_values=0)
            Y_solvAcc = np.pad(Y_solvAcc, pad_width=(0, n_missing), mode='constant', constant_values=0)
            Y_flex = np.pad(Y_flex, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_struct = np.pad(mask_struct, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_solvAcc = np.pad(mask_solvAcc, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask_flex = np.pad(mask_flex, pad_width=(0, n_missing), mode='constant', constant_values=0)
            X = np.pad(X, pad_width=((0, n_missing), (0, 0)), mode='constant', constant_values=0.)

        X = X.T
        X = np.expand_dims(X, axis=1)
        Y_struct = np.expand_dims(Y_struct, axis=1)
        Y_solvAcc = np.expand_dims(Y_solvAcc, axis=1)
        Y_flex = np.expand_dims(Y_flex, axis=1)
        mask_struct = np.expand_dims(mask_struct, axis=0)
        mask_solvAcc = np.expand_dims(mask_solvAcc, axis=0)
        mask_flex = np.expand_dims(mask_flex, axis=0)
        Y_struct = Y_struct.T
        Y_solvAcc=Y_solvAcc.T
        Y_flex = Y_flex.T


        input_tensor = torch.from_numpy(X).type(dtype=torch.float)
        struct_tensor = torch.from_numpy(Y_struct)
        solvAcc_tensor=torch.from_numpy(Y_solvAcc)
        flex_tensor = torch.from_numpy(Y_flex)
        mask_struct_tensor = torch.from_numpy(mask_struct).type(dtype=torch.float)
        mask_solvAcc_tensor = torch.from_numpy(mask_solvAcc).type(dtype=torch.float)
        mask_flex_tensor = torch.from_numpy(mask_flex).type(dtype=torch.float)

        weighted_mask_struct_tensor=torch.Tensor(np.array([self.weights[0][i] for i in struct_tensor]))*mask_struct_tensor
        #weighted_mask_solvAcc_tensor=torch.Tensor(np.array([self.weights[1][i] for i in solvAcc_tensor]))*mask_solvAcc_tensor
        #weighted_mask_flex_tensor=torch.Tensor(np.array([self.weights[2][i] for i in flex_tensor]))*mask_flex_tensor
        #assert weighted_mask_struct_tensor.size()==mask_struct_tensor.size()

        return (input_tensor, struct_tensor, solvAcc_tensor, flex_tensor, weighted_mask_struct_tensor, mask_solvAcc_tensor, mask_flex_tensor)

    def __len__(self):
        return self.data_len


#
# Returns train set and test set
#
def train_val_test_split(inputs, targets_struct, targets_solcAcc, targets_flex, masks_struct, masks_solvAcc, masks_flex,weights):
    d = Data_splitter(inputs, targets_struct, targets_solcAcc, targets_flex, masks_struct, masks_solvAcc, masks_flex)
    train, val, test = d.split_data()
    max_len = d.get_max_length()
    train_set = Dataset(train, weights,max_len)
    val_set = Dataset(val,weights, max_len)

    return train_set, val_set

#
# Creates Loader and TrainLoader of specified batchsizes
#
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