#!/usr/bin/python3 -u

import numpy as np
import os
import sys

save_path='/mnt/home/reithmeir/datapreprocessing'


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

def readStructures3(protein):
    path = '/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp/' + protein.lower() + '/' + protein.lower() + '.3.consensus.dssp'
    print('struct3 path:',path)
    if(os.path.isfile(path)):
        f = open(path, 'r')
        structures_input = f.read().splitlines()
        f.close()
        structures=''
        for i in range(len(structures_input)):
            if(i>0):
                structures += structures_input[i]
        return structures
    else:
        print('no 3 file of', protein)

def readStructures8(protein):
    path = '/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp/' + protein.lower() + '/' + protein.lower() + '.8.consensus.dssp'
    print('struct8 path:', path)
    if(os.path.isfile(path)):
        f = open(path, 'r')
        structures_input = f.read().splitlines()
        f.close()
        structures = ''
        for i in range(len(structures_input)):
            if (i > 0):
                structures += structures_input[i]
        return structures
    else:
        print('no 8 file of ', protein)


proteinPath = sys.argv[1]
proteinName=proteinPath.split('/')[-1].split('.')[0]
if(not os.path.isdir(save_path)):
    os.mkdir(save_path)
if(not os.path.isdir(save_path+'/'+proteinName.upper())):
    os.mkdir(save_path+'/'+proteinName.upper())

res = readResidues(proteinPath)
str3 = readStructures3(proteinName)
str8 = readStructures8(proteinName)
if(str3):
    struct_memmap_3 = [DSSP3_MAPPING[dssp_state] for dssp_state in str3]
    mask = [0 if dssp_state in DSSP3_NAN else 1 for dssp_state in str3]
    np.save(save_path + '/' + proteinName.upper() + '/mask_3.npy', np.array(mask, dtype=int))
    np.save(save_path+'/'+proteinName.upper()+'/structures_3.npy', np.array(struct_memmap_3, dtype=int))

if(str8):
    struct_memmap_8 = [DSSP8_MAPPING[dssp_state] for dssp_state in str8]
    mask = [0 if dssp_state in DSSP3_NAN else 1 for dssp_state in str8]
    np.save(save_path + '/' + proteinName.upper() + '/mask_8.npy', np.array(mask, dtype=int))
    np.save(save_path + '/' + proteinName.upper() + '/structures_8.npy', np.array(struct_memmap_8, dtype=int))