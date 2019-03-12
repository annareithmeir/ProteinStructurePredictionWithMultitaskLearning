#!/usr/bin/python3 -u

import numpy as np
import os
import sys

#
# This file creates the ProtVec representation for a given protein
#


save_path='/mnt/home/reithmeir/datapreprocessing'
protvec_path='/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/protVecs/protVec_100d_3grams.csv'


def readResidues(path): # Gets sequence of residues for protein
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

def createProtVec(seq): # Creates the Protvec representation
    protVec_repr = np.zeros((len(seq), 100))
    for i in range(len(seq)-2):
        if (seq[i] in ['X','O'] or seq[i+1] in ['X','O'] or seq[i+2] in ['X','O']):
            print('Non standard protein at ', i, ' which is ', seq[i:i+3])

        else:
            if (i == len(seq) - 2):
                protVec_repr[i + 1] = PROTVECS[seq[-3:]]
            else:
                protVec_repr[i + 1] = PROTVECS[seq[i:i + 3]]
    return protVec_repr

def protVec2dict(): # Reads the ProtVec vectors into a dict
    protVec = np.genfromtxt(protvec_path, dtype=str)
    dict={}
    for line in protVec :
        if(line[0]!='words'):
            tmp={line[0] : np.array(line[1:]).astype(np.float)}
            dict.update(tmp)
    print('protvecs:',dict)
    return dict

PROTVECS=protVec2dict()

proteinPath = sys.argv[1]
res = readResidues(proteinPath)
protvec=createProtVec(res)
proteinName=proteinPath.split('/')[-1].split('.')[0]

if(not os.path.isdir(save_path)):
    os.mkdir(save_path)
if(not os.path.isdir(save_path+'/'+proteinName.upper())):
    os.mkdir(save_path+'/'+proteinName.upper())

np.save(save_path+'/'+proteinName.upper()+'/protvec.npy', np.array(protvec))

