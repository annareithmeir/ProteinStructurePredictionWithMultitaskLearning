import numpy as np
import pandas as pd
import os

proteins_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp'
seq_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences'

DSSP3_MAPPING = { 'C' : 0,
                  'H' : 1,
                  'E' : 2,
                  'Y' : 3, # map ambigious residues also to E, mask out later
                  'X' : 3  # map missing residues also to E, mask out later
                  }

def readResidues(protein):
    path=seq_path+'/'+protein.upper()+'.fasta.txt'
    if(os.path.isfile(path)):
    	residues_input = np.loadtxt(path, dtype=str)
    	residues=''
    	for i in range(len(residues_input)):
            if(i>0):
            	if(residues_input[i][0]=='>'):
                	break
            	residues += residues_input[i]

    	return residues
    else:
	print('no residue')

def readStructures3(protein):
    path= proteins_path+'/'+protein+'/'+protein+'.3.consensus.dssp'
    if(os.path.isfile(path)):
        structures_input = np.loadtxt(path, dtype=str)
        structures=''
        for i in range(len(structures_input)):
	    if(i>0):
	        structures += structures_input[i]
	return structures
    else:
        print('no 3 file of', protein)

def readStructures8(protein):
    path = proteins_path+'/'+protein+'/'+protein+'.8.consensus.dssp'
    if(os.path.isfile(path)):
        structures_input = np.loadtxt(path, dtype=str)
	structures = ''
        for i in range(len(structures_input)):
       	    if (i > 0):
		structures += structures_input[i]
	return structures
    else:
        print('no 8 file of ', protein)

def create1hot(sequence):
    length = len(sequence)
    matrix = np.zeros((length, 20), dtype=int)
    onehot = pd.DataFrame(matrix,
                          columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W', 'Y', 'V'],
                          index=[np.arange(length)])
    for i in range(length):
        if(sequence[i] in onehot.keys()):
            onehot[sequence[i]][i] = 1
        else:
            print('None standard protein at ',i,' which is ',sequence[i])

    return onehot  # returns dataframe. to have it as np array use onehot.values

def getInput(folder, classification_mode): #proteins in a folder named root+proteins/
    dict={}
    for prot in os.listdir(folder):
        res=readResidues(prot)
        if(classification_mode==3):
            str=readStructures3(prot)
        elif(classification_mode==8):
            str=readStructures8(prot)
        else:
            raise ValueError('[ERROR] Either 3 or 8 classes')
	if(str!=None and res!=None):
            tmp = {prot: (res, str)}
            dict.update(tmp)
    return dict

def countStructures3(sequence):
    c=0
    h=0
    e=0
    xy=0
    for i in range(len(sequence)):
        if(sequence[i]=='C'):
            c+=1
        elif(sequence[i]=='H'):
            h+=1
        elif(sequence[i]=='E'):
            e+=1
        elif(sequence[i]=='X' or sequence[i]=='Y'):
            xy+=1
        else:
            raise ValueError('Unknown structure')

    return c,h,e,xy






dict = getInput(proteins_path, 3)
matrix_train=[]
targets_train=[]
matrix_val=[]
targets_val=[]

np.random.seed(42) #to reproduce splits
val_dict={}
tmp_dict={}
c_train = 0
h_train = 0
e_train = 0
xy_train = 0
c_val = 0
h_val = 0
e_val = 0
xy_val = 0

for i in range(len(dict)):
    rnd = np.random.rand()
    seq = dict[dict.keys()[i]][0]
    m = create1hot(seq).values
    struct = dict[dict.keys()[i]][1]
    struct_memmap = [DSSP3_MAPPING[dssp_state] for dssp_state in struct]
    tmp_c, tmp_h, tmp_e, tmp_xy = countStructures3(struct)
    if (rnd > 0.8):
        matrix_val.append(m)
        targets_val.append(struct_memmap)
        c_val+=tmp_c
        h_val+=tmp_h
        e_val+=tmp_e
        xy_val+=tmp_xy
    else:
        matrix_train.append(m)
        targets_train.append(struct_memmap)
        c_train += tmp_c
        h_train += tmp_h
        e_train += tmp_e
        xy_train += tmp_xy

sum=float(c_val+h_val+e_val+xy_val)
print('DICT ---- C: ', c_val/sum, ' , H: ', h_val/sum, ' , E: ', e_val/sum, ' ,XY: ', xy_val/sum, 'sum: ', sum)
sum=float(c_train+h_train+e_train+xy_train)
print('TEST_DICT ---- C: ', c_train/sum, ' , H: ', h_train/sum, ' , E: ', e_train/sum, ' ,XY: ', xy_train/sum, 'sum: ', sum)

matrix_val_np=np.array(matrix_val)
print(matrix_val_np.shape)
targets_val_np=np.array(targets_val)
print(targets_val_np.shape)
np.save('matrix_1hot_3_val.npy', matrix_val_np)
np.save('targets_val.npy',targets_val_np)

matrix_train_np=np.array(matrix_train)
print(matrix_train_np.shape)
targets_train_np=np.array(targets_train)
print (targets_train_np.shape)
np.save('matrix_1hot_3_train.npy', matrix_train_np)
np.save('targets_train.npy',targets_train_np)
print('done')
