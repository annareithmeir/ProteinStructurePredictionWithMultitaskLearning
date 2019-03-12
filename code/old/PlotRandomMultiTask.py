import numpy as np
import os
import torch
import torch.nn as nn
import Networks

from random import randint



LOG_PATH = 'log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask'
INPUT_PATH='preprocessing'
TARGETS_PATH='/home/mheinzinger/contact_prediction_v2/targets'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=Networks.DenseHybrid('protvec+scoringmatrix', 3, 'multi4', 6, 1, 1,device).to(device)
model.load_state_dict(torch.load(LOG_PATH+'/cnn_multitask.ckpt'))
model.eval()

protein_idx=randint(0,2935)

files=[]
for (dirpath, dirnames, filenames) in os.walk('dataset_preprocessed'):
    files.extend(dirnames)
    break

protein=files[protein_idx]
print(protein)

seq = np.load(INPUT_PATH + '/' + protein + '/protvec+scoringmatrix.npy')
seq=torch.Tensor(seq)
seq=seq.to(device)
print(seq.size())

flex = np.memmap(TARGETS_PATH + '/bdb_bvals/' + protein.lower() + '.bdb.memmap', dtype=np.float32, mode='r', shape=len(seq))
flex = np.nan_to_num(flex)

mask_flex = np.ones(len(flex))
nans = np.argwhere(np.isnan(flex.copy()))
mask_flex[nans] = 0

solvAcc = np.memmap(TARGETS_PATH + '/dssp/' + protein.lower() + '/' + protein.lower() + '.rel_asa.memmap',dtype=np.float32, mode='r', shape=len(seq))
msolvAcc = np.nan_to_num(solvAcc)

mask_solvAcc = np.ones(len(seq))
nans = np.argwhere(np.isnan(solvAcc.copy()))
mask_solvAcc[nans] = 0

struct3_pred, struct8_pred, solvacc_pred, flex_pred=model(seq)



