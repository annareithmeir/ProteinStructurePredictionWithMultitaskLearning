import numpy as np
import pandas as pd
import os

#proteins_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp'
proteins_path='../mheinzinger/contact_prediction_v2/targets/dssp'
#seq_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences'
seq_path='../mheinzinger/contact_prediction_v2/sequences'
#protvec_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/protVecs/protVec_100d_3grams.csv'
protvec_path='../mheinzinger/contact_prediction_v2/protVecs/protVec_100d_3grams.csv'
save_path='matrices_012'

DSSP3_MAPPING = { 'C' : 0,
                  'H' : 1,
                  'E' : 2,
                  'Y' : 2, # map ambigious residues also to E, mask out later
                  'X' : 2  # map missing residues also to E, mask out later
                  }

DSSP8_MAPPING = { 'H' : 0,
                  'E' : 1,
                  'I' : 2,
                  'S' : 3,
                  'T' : 4,
                  'G' : 5,
                  'B' : 6,
                  '-' : 7,
                  'Y' : 7, # map ambigious residues also to E, mask out later
                  'X' : 7  # map missing residues also to E, mask out later
                  }

DSSP3_NAN = { 'X', 'Y' }

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
            print('Non standard protein at ',i,' which is ',sequence[i])

            with open("results.txt", 'a') as results:
                results.write("\n ######### creating 1-hot ########## \n")
                results.write(str('Non standard protein at '+str(i)+' which is '+sequence[i]))
                results.write("\n #################################### \n")

    return onehot  # returns dataframe. to have it as np array use onehot.values

def protVec2dict():
    protVec = np.genfromtxt(protvec_path, dtype=str)
    dict={}
    for line in protVec :
        if(line[0]!='words'):
            tmp={line[0] : np.array(line[1:]).astype(np.float)}
            dict.update(tmp)
    return dict

def createProtVec(seq):
    protVec_repr = np.zeros((len(seq), 100))
    for i in range(len(seq)-2):
        if (seq[i] in ['X','O'] or seq[i+1] in ['X','O'] or seq[i+2] in ['X','O']):
            print('Non standard protein at ', i, ' which is ', seq[i:i+3])

            with open("results.txt", 'a') as results:
                results.write("\n ######### creating ProtVec ########## \n")
                results.write(str('Non standard protein at ' + str(i) + ' which is ' + seq[i:i+3]))
                results.write("\n #################################### \n")
        else:
            if (i == len(seq) - 2):
                protVec_repr[i + 1] = PROTVECS[seq[-3:]]
            else:
                protVec_repr[i + 1] = PROTVECS[seq[i:i + 3]]
    return protVec_repr

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




PROTVECS=protVec2dict()

dict_3 = getInput(proteins_path, 3)
dict_8 = getInput(proteins_path, 8)
matrix_train=[]
matrix_val=[]
protvec_train=[]
protvec_val=[]
targets_val_3=[]
targets_val_8=[]
targets_train_3=[]
targets_train_8=[]
masks_train=[]
masks_val=[] # TODO: check if mask is same for dssp3 and dssp8!

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


for tmp in dict_3.keys():
    print('tmp:',tmp)
    rnd = np.random.rand()
    seq = dict_3[tmp][0]
    m = create1hot(seq).values
    pv=createProtVec(seq)
    struct_3 = dict_3[tmp][1]
    struct_8 = dict_8[tmp][1]
    struct_memmap_3 = [DSSP3_MAPPING[dssp_state] for dssp_state in struct_3]
    struct_memmap_8 = [DSSP8_MAPPING[dssp_state] for dssp_state in struct_8]
    mask=[ 0 if dssp_state in DSSP3_NAN else 1 for dssp_state in struct_3 ]
    tmp_c, tmp_h, tmp_e, tmp_xy = countStructures3(struct_3)
    if (rnd > 0.8):
        matrix_val.append(m)
        protvec_val.append(pv)
        targets_val_3.append(struct_memmap_3)
        targets_val_8.append(struct_memmap_8)
        masks_val.append(mask)
        c_val+=tmp_c
        h_val+=tmp_h
        e_val+=tmp_e
        xy_val+=tmp_xy
    else:
        matrix_train.append(m)
        protvec_train.append(pv)
        targets_train_3.append(struct_memmap_3)
        targets_train_8.append(struct_memmap_8)
        masks_train.append(mask)
        c_train += tmp_c
        h_train += tmp_h
        e_train += tmp_e
        xy_train += tmp_xy

sum=float(c_val+h_val+e_val+xy_val)
print('sum', sum)
print('DICT ---- C: ', c_val/sum, ' , H: ', h_val/sum, ' , E: ', e_val/sum, ' ,XY: ', xy_val/sum, 'sum: ', sum)
sum=float(c_train+h_train+e_train+xy_train)
print('sum', sum)
print('TEST_DICT ---- C: ', c_train/sum, ' , H: ', h_train/sum, ' , E: ', e_train/sum, ' ,XY: ', xy_train/sum, 'sum: ', sum)

matrix_val_np=np.array(matrix_val)
print('matrix val shape=',matrix_val_np.shape)
protvec_val_np=np.array(protvec_val)
print('protvec val shape=',protvec_val_np.shape)
targets_val_3_np=np.array(targets_val_3)
print('targets 3 val shape=',targets_val_3_np.shape)
targets_val_8_np=np.array(targets_val_8)
print('targets 8 val shape=',targets_val_8_np.shape)
masks_val_np=np.array(masks_val)
print('masks val shape=',masks_val_np.shape)
np.save(save_path+'/matrix_1hot_val.npy', matrix_val_np)
np.save(save_path+'/matrix_protvec_val.npy', protvec_val_np)
np.save(save_path+'/targets_3_val.npy',targets_val_3_np)
np.save(save_path+'/targets_8_val.npy',targets_val_8_np)
np.save(save_path+'/masks_3_val.npy',masks_val_np)

matrix_train_np=np.array(matrix_train)
print('matrix train shape=',matrix_train_np.shape)
protvec_train_np=np.array(protvec_train)
print('protvec train shape=',protvec_train_np.shape)
targets_train_3_np=np.array(targets_train_3)
print ('targets train 3 shape=',targets_train_3_np.shape)
targets_train_8_np=np.array(targets_train_8)
print ('targets train 8 shape=',targets_train_8_np.shape)
masks_train_np=np.array(masks_train)
print('masks train shape=',masks_train_np.shape)
np.save(save_path+'/matrix_1hot_train.npy', matrix_train_np)
np.save(save_path+'/matrix_protvec_train.npy', protvec_train_np)
np.save(save_path+'/targets_3_train.npy',targets_train_3_np)
np.save(save_path+'/targets_8_train.npy',targets_train_8_np)
np.save(save_path+'/masks_3_train.npy',masks_train_np)
print('done')
