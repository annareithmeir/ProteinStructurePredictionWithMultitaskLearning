#!/usr/bin/python3 -u
import numpy as np
import sys
import os

#
# This file parses the PSIBLAST generated PSSM matrix of one protein and generates
# The ProtVec representation concatenated to the PSSM matrix
#


load_path='/home/areithmeier/PSSM_all'

def parsePSSM(protein):
    f = open(load_path + '/' + protein + '_pssm.txt', 'r')
    input = f.read().splitlines()
    f.close()

    scoringmatrix = np.zeros((len(input) - 7, 20))
    percentages = np.zeros((len(input) - 7, 20))
    information = np.zeros((len(input) - 7, 1))
    gapweights = np.zeros((len(input) - 7, 1))

    for j in range(len(input) - 4):
        if (j > 2):
            for k in range(20):
                scoringmatrix[j - 3][k] = int(input[j][11:].split()[k])
                percentages[j - 3][k] = int(input[j][11:].split()[k + 20])
            information[j - 3] = float(input[j][11:].split()[40])
            gapweights[j - 3] = float(input[j][11:].split()[41])

    protvec = np.load('/home/areithmeier/preprocessing/' + protein.upper() + '/protvecevolutionary.npy')
    protvec_matrix = np.concatenate((protvec, scoringmatrix), axis=1)
    protvec_information = np.concatenate((protvec, information), axis=1)
    protvec_gapweights = np.concatenate((protvec, gapweights), axis=1)

    assert (protvec_matrix.shape[1] == 120)
    assert (protvec_information.shape[1] == 101)
    assert (protvec_gapweights.shape[1] == 101)

    np.save('preprocessing/' + protein.upper() + '/protvec+scoringmatrix.npy', np.array(protvec_matrix))


protein = sys.argv[1]
if(os.path.isfile(save_path+'/'+ protein+'_pssm.txt')):
    parsePSSM(protein)
else:
    print('no pssm file')

