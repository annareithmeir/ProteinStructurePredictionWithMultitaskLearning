#!/usr/bin/python3 -u

import numpy as np
import os
import sys

#
# This file generates the ProtVec representation of the MSA of a given protein
#


save_path='/mnt/home/reithmeir/datapreprocessing'
protvec_path='/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/protVecs/protVec_100d_3grams.csv'


def readResidues(path): # Reads sequence of residues for protein
    if(os.path.isfile(path)):
        f = open(path, 'r')
        residues_input = f.read().splitlines()
        f.close()
        residues=''
        for i in range(len(residues_input)):
            if(i>0):
                if(residues_input[i][0]=='>'):
                    break
                residues += residues_input[i]

        return residues
    else:
        print('no residue')

def createProtVecEvolutionarySequence(seq): # Creates ProtVec representation for one sequence
    NONRESIDUES=['X','O','-']
    positives = np.zeros(len(seq))
    protVec_repr = np.zeros((len(seq), 100))
    for i in range(len(seq) - 2):
        if (seq[i] in NONRESIDUES or seq[i + 1] in NONRESIDUES or seq[i + 2] in NONRESIDUES):
            tmp= np.zeros(100)
            protVec_repr[i + 1] =tmp

        else:
            if (i == len(seq) - 2):
                protVec_repr[i + 1] = PROTVECS[seq[-3:]]
                positives[i + 1] += 1
            else:
                try:
                    protVec_repr[i + 1] = PROTVECS[seq[i:i + 3]]
                    positives[i + 1] += 1
                except(KeyError):
                    print('KEY ERROR:',seq[i:i + 3])
                    tmp = np.zeros(100)
                    protVec_repr[i + 1] = tmp
    assert (positives[0] == 0 and positives[-1] == 0)
    return protVec_repr, positives

def readAlignments(path): #Reads MSA for a protein
    f = open(path, 'r')
    alignments_input = f.read().splitlines()
    f.close()
    alignments=[]
    for i in range(len(alignments_input)):
        alignments.append(alignments_input[i])
    return alignments

def createProtVecEvolutionary(alignments): #Creates ProtVec representation for MSA
    protvec_avg=np.zeros((len(alignments[0]),100))
    divisors=np.zeros(len(alignments[0]))

    for seq in alignments:
        protvec, positives=createProtVecEvolutionarySequence(seq)
        protvec_avg+=protvec
        divisors+=positives

    divisors[1:-1]=1/divisors[1:-1]
    return np.multiply(protvec_avg, divisors[:,np.newaxis])

def protVec2dict(): #Reads ProtVec vectors to dict
    protVec = np.genfromtxt(protvec_path, dtype=str)
    dict={}
    for line in protVec :
        if(line[0]!='words'):
            tmp={line[0] : np.array(line[1:]).astype(np.float)}
            dict.update(tmp)
    return dict

PROTVECS=protVec2dict()

proteinPath = sys.argv[1]
proteinName=proteinPath.split('/')[-1].split('.')[0]
alignmentsPath='/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments/'+proteinName.upper()+'/'+proteinName.upper()+'.psicov'
if (not os.path.isdir(save_path)):
    os.mkdir(save_path)
if (not os.path.isdir(save_path + '/' + proteinName.upper())):
    os.mkdir(save_path + '/' + proteinName.upper())

if(os.path.isfile(alignmentsPath)):
    al = readAlignments(alignmentsPath)
    protvec=createProtVecEvolutionary(al)
    proteinName=alignmentsPath.split('/')[-1].split('.')[0]

    np.save(save_path+'/'+proteinName.upper()+'/protvec_evolutionary.npy', np.array(protvec))

