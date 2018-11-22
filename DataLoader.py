import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import utilities_MICHAEL as utilities


def countStructures3(train, test):
    def countStructs(dict):
        c = 0
        h = 0
        e = 0
        xy = 0
        for sample in dict.keys():
            sequence = dict[sample][1]
            for res in range(len(sequence)):
                if (sequence[res] == 1):
                    c += 1
                elif (sequence[res] == 2):
                    h += 1
                elif (sequence[res] == 3):
                    e += 1
                elif (sequence[res] == 4):
                    xy += 1
                else:
                    raise ValueError('Unknown structure', sequence[res])

        return c, h, e, xy

    # check if classes occur around same times in train and test set
    ctrain, htrain, etrain, xytrain = countStructures3(train)
    trainsum = ctrain + htrain + etrain + xytrain
    ctest, htest, etest, xytest = countStructures3(test)
    testsum = ctest + htest + etest + xytest
    print('training structure ratio:', ctrain / float(trainsum), htrain / float(trainsum), etrain / float(trainsum),
          xytrain / float(trainsum))
    print('occurences train:', ctrain, htrain, etrain, xytrain, '-->', trainsum)
    print('testing structure ratio:', ctest / float(testsum), htest / float(testsum), etest / float(testsum),
          xytest / float(testsum))
    print('occurences test:', ctest, htest, etest, xytest, '-->', testsum)

class Data_splitter():
    def __init__(self, inputs, targets, test_ratio=0.2):  # 80% train, 20% test
        self.inputs = inputs
        self.targets = targets
        self.test_ratio = test_ratio
        self.max_len = 0

    def split_data(self):
        train = dict()
        test = dict()
        cnt_test = 0
        cnt_train = 0
        np.random.seed(42)  # to reproduce splits
        for i in range(len(self.inputs)):
            rnd = np.random.rand()

            if (len(self.targets[i]) > self.max_len):
                self.max_len = len(self.targets[i])
            if (rnd > (1 - self.test_ratio)):
                test[cnt_test] = (self.inputs[i], self.targets[i])
                cnt_test += 1
            else:
                train[cnt_train] = (self.inputs[i], self.targets[i])
                cnt_train += 1

        print('Size of training set: {}'.format(len(train)))
        print('Size of test set: {}'.format(len(test)))

        return train, test

    def get_max_length(self):
        return self.max_len

class Dataset(data.Dataset):
    # LEN_FEATURE_VEC = 1

    def __init__(self, samples, max_len=None):
        self.max_len = max_len
        self.inputs, self.targets = zip(*[(inputs, targets)
                                          for _, (inputs, targets) in samples.items()])
        self.data_len = len(self.inputs)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        X = self.inputs[idx]
        # print('X.shape in getitem(): ', X.shape)
        Y = self.targets[idx]
        # print('Y.shape in getitem(): ', len(Y))
        mask_idx = np.where(np.array(Y) == 4)
        mask = np.ones(len(X))
        mask[mask_idx] = 0
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
        # print('inputtensor:', input_tensor.size())
        output_tensor = torch.from_numpy(Y)
        # print('outputtensor:', output_tensor.size())
        mask_tensor = torch.from_numpy(mask).type(dtype=torch.float)
        # print('masktensor', mask_tensor.size())
        return (input_tensor, output_tensor, mask_tensor)

    def __len__(self):
        return self.data_len

def train_test_split(inputs, targets):
    d = Data_splitter(inputs, targets)
    train, test = d.split_data()
    max_len = d.get_max_length()
    train_set = Dataset(train, max_len)
    test_set = Dataset(test, max_len)

    return train_set, test_set

def createDataLoaders(train_set, test_set):
    data_loaders = dict()
    data_loaders['Train'] = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=32,
                                                        shuffle=True,
                                                        drop_last=False
                                                        )
    data_loaders['Test'] = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=100,
                                                       shuffle=False,
                                                       drop_last=False
                                                       )
    return data_loaders