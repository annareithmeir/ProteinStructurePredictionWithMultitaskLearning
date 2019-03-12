import numpy as np
import os
import shutil

TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'

files=[]
for (dirpath, dirnames, filenames) in os.walk('testset_preprocessed'):
    files.extend(dirnames)
    break

with open('/home/areithmeier/log/multi3/3/DenseHybrid_protvec+scoringmatrix_3_multitask/multitask_log.txt', 'r') as file:
    data = file.read().split('\n')

    truesa = data[data.index('truesa') + 1:data.index('predfl')]
    truesa = list(map(float, truesa))

    truefl = data[data.index('truefl') + 1:-1]
    truefl = list(map(float, truefl))


    print(truefl)


for protein in files:
    flex = np.memmap(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap', dtype=np.float32, mode='r')

    maskfl = np.ones(len(flex))
    nans = np.argwhere(np.isnan(flex.copy()))
    maskfl[nans] = 0
    flex = flex[maskfl != 0]
    flex=map(float,flex)

    '''
    solvAcc = np.memmap(TARGETS_PATH + '/dssp/' + protein.lower() + '/' + protein.lower() + '.rel_asa.memmap',
                        dtype=np.float32, mode='r')

    masksa = np.ones(len(flex))
    nans = np.argwhere(np.isnan(solvAcc.copy()))
    masksa[nans] = 0
    solvAcc = solvAcc[masksa != 0]
    '''

    if(round(flex[0],3)==round(truefl[0],3) and
            round(flex[1], 3) == round(truefl[1], 3) and
            round(flex[2], 3) == round(truefl[2], 3)
    ):
        print(protein)
        print(flex)
        print(truefl)

