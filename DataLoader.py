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
def countStructures3(train, test):
    def countStructs(dict):
        c = 0
        h = 0
        e = 0
        for sample in dict.keys():
            sequence = dict[sample][1]
            mask=dict[sample][2]
            for res in range(len(sequence)):
                if (sequence[res] == 0 and mask[res]!=0):
                    c += 1
                elif (sequence[res] == 1 and mask[res]!=0):
                    h += 1
                elif (sequence[res] == 2 and mask[res]!=0):
                    e += 1
                elif (mask[res]==0):
                    pass
                else:
                    raise ValueError('Unknown structure', sequence[res])

        return c, h, e

    # check if classes occur around same times in train and test set
    ctrain, htrain, etrain = countStructs(train)
    trainsum = ctrain + htrain + etrain
    ctest, htest, etest = countStructs(test)
    testsum = ctest + htest + etest
    print('training structure ratio %:', round(ctrain / float(trainsum)*100,2), round(htrain / float(trainsum)*100,2), round(etrain / float(trainsum)*100,2))
    #print('occurences train:', ctrain, htrain, etrain, '-->', trainsum)
    print('testing structure ratio %:', round(ctest / float(testsum)*100,2), round(htest / float(testsum)*100,2), round(etest / float(testsum)*100,2))
    #print('occurences test:', ctest, htest, etest, '-->', testsum)

#
# Splits data into training and testing set
#
class Data_splitter():
    def __init__(self, inputs, targets, masks, test_ratio=0.2):  # 80% train, 20% test
        self.inputs = inputs
        self.targets = targets
        self.masks=masks
        self.test_ratio = test_ratio
        self.max_len = 0

    def split_data(self):
        train = dict()
        test = dict()
        cnt_test = 0
        cnt_train = 0
        np.random.seed(42)  # to reproduce splits
        proteins=self.inputs.keys()
        for i in proteins:
            rnd = np.random.rand()

            if (len(self.targets[i]) > self.max_len):
                self.max_len = len(self.targets[i])
            if (rnd > (1 - self.test_ratio)):
                test[cnt_test] = [self.inputs[i], self.targets[i], self.masks[i]]
                cnt_test += 1
            else:
                train[cnt_train] = [self.inputs[i], self.targets[i], self.masks[i]]
                cnt_train += 1

        print('Size of training set: {}'.format(len(train)))
        print('Size of test set: {}'.format(len(test)))
        countStructures3(train, test)

        return train, test

    def get_max_length(self):
        return self.max_len


#
# Custom dataset class, returns (ProtVec/1hot, target, weighted mask) for one sample.
# Each sample is padded with zeros and has length 968
#
class Dataset(data.Dataset):

    def __init__(self, samples, weights, max_len=None):
        self.max_len = max_len
        self.inputs, self.targets, self.masks = zip(*[[inputs, targets, masks]
                                          for _, [inputs, targets, masks] in samples.items()])
        self.data_len = len(self.inputs)
        self.to_tensor = transforms.ToTensor()
        self.weights=weights

    def __getitem__(self, idx):
        X = self.inputs[idx]
        # print('X.shape in getitem(): ', X.shape)
        Y = self.targets[idx]
        # print('Y.shape in getitem(): ', len(Y))
        mask=self.masks[idx]
        #mask_idx = np.where(np.array(Y) == 4)
        #mask = np.ones(len(X))
        #mask[mask_idx] = 0
        # print('Y:', Y)
        # print('mask.shape in getitem(): ', mask.shape)

        if self.max_len:  # if a maximum length (longest seq. in the set) was provided
            n_missing = self.max_len - len(
                Y)  # get the number of residues which have to be padded to achieve same size for all proteins in the set
            Y = np.pad(Y, pad_width=(0, n_missing), mode='constant', constant_values=0)
            mask = np.pad(mask, pad_width=(0, n_missing), mode='constant', constant_values=0)
            X = np.pad(X, pad_width=((0, n_missing), (0, 0)), mode='constant', constant_values=0.)

        X = X.T
        # print('Xt.shape in getitem(): ', X.shape)
        X = np.expand_dims(X, axis=1)
        # print('Xt_expanded.shape in getitem(): ', X.shape)
        Y = np.expand_dims(Y, axis=1)
        # print('Y_expanded.shape in getitem(): ', Y.shape)
        mask = np.expand_dims(mask, axis=0)
        # print('mask_expanded.shape in getitem(): ', mask.shape)
        Y = Y.T
        # print('Yt_expanded.shape in getitem(): ', Y.shape)

        input_tensor = torch.from_numpy(X).type(dtype=torch.float)
        #print('inputtensor:', input_tensor.size(), input_tensor)
        output_tensor = torch.from_numpy(Y)
        #print('outputtensor:', output_tensor.size(), output_tensor)
        mask_tensor = torch.from_numpy(mask).type(dtype=torch.float)
        #print('masktensor', mask_tensor.size(), mask_tensor)

        weighted_mask=torch.Tensor(np.array([self.weights[i] for i in output_tensor]))*mask_tensor
        #print('weighted:', weighted_mask)
        assert weighted_mask.size()==mask_tensor.size()

        return (input_tensor, output_tensor, weighted_mask)

    def __len__(self):
        return self.data_len


#
# Returns train set and test set
#
def train_test_split(inputs, targets, masks, weights):
    d = Data_splitter(inputs, targets, masks)
    train, test = d.split_data()
    max_len = d.get_max_length()
    train_set = Dataset(train, weights,max_len)
    test_set = Dataset(test,weights, max_len)

    return train_set, test_set

#
# Creates TestLoader and TrainLoader of specified batchsizes
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