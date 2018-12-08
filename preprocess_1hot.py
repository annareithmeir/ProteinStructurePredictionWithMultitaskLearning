#!/usr/bin/python3 -u

import numpy as np
import os
import sys

save_path='/mnt/home/reithmeir/datapreprocessing'

def readResidues(path):
    if(os.path.isfile(path)):
        f = open(path, 'r')
        residues_input = f.read().splitlines()
        f.close()
        print('resin:' ,residues_input)
        residues=''
        for i in range(len(residues_input)):
            if(i>0):
                if(residues_input[i][0]=='>'):
                    break
                residues += residues_input[i]

        return residues
    else:
        print('no residue')

def create1hot(sequence):
    length = len(sequence)
    onehot = np.zeros((20,length), dtype=int)
    #onehot = pd.DataFrame(matrix,
    #                      columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
    #                               'W', 'Y', 'V'],
    #                      index=[np.arange(length)])
    for i in range(length):
        if(sequence[i] is 'A'):
            onehot[0][i] = 1
        elif(sequence[i] is 'R'):
            onehot[1][i] =1
        elif(sequence[i] is 'N'):
            onehot[2][i] =1
        elif(sequence[i] is 'D'):
            onehot[3][i] = 1
        elif(sequence[i] is 'C'):
            onehot[4][i] = 1
        elif(sequence[i] is 'E'):
            onehot[5][i] = 1
        elif(sequence[i] is 'Q'):
            onehot[6][i] = 1
        elif(sequence[i] is 'G'):
            onehot[7][i] = 1
        elif(sequence[i] is 'H'):
            onehot[8][i] = 1
        elif(sequence[i] is 'I'):
            onehot[9][i] = 1
        elif(sequence[i] is 'L'):
            onehot[10][i] = 1
        elif(sequence[i] is 'K'):
            onehot[11][i] =1
        elif(sequence[i] is 'M'):
            onehot[12][i] =1
        elif(sequence[i] is 'F'):
            onehot[13][i] = 1
        elif(sequence[i] is 'P'):
            onehot[14][i] = 1
        elif(sequence[i] is 'S'):
            onehot[15][i] = 1
        elif(sequence[i] is 'T'):
            onehot[16][i] = 1
        elif(sequence[i] is 'W'):
            onehot[17][i] = 1
        elif(sequence[i] is 'Y'):
            onehot[18][i] = 1
        elif(sequence[i] is 'V'):
            onehot[19][i] = 1
        else:
            print('Non standard protein at ',i,' which is ',sequence[i])

    return onehot.T  # returns dataframe. to have it as np array use onehot.values

proteinPath = sys.argv[1]
print(proteinPath)
res = readResidues(proteinPath)
print(res)
onehot=create1hot(res)
proteinName=proteinPath.split('/')[-1].split('.')[0]
print(proteinName)

if(not os.path.isdir(save_path)):
    os.mkdir(save_path)
if(not os.path.isdir(save_path+'/'+proteinName.upper())):
    os.mkdir(save_path+'/'+proteinName.upper())

np.save(save_path+'/'+proteinName.upper()+'/1hot.npy', np.array(onehot))