import numpy as np
import pandas as pd
import os

proteins_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp'
seq_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences'

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
            tmp={prot : (res, str)}
            dict.update(tmp)
    return dict


### ANALYSE THE DATA ###
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
            raise ValueError('Unknown structure', sequence[i])

    #print('####')
    #print('C: '+str(c))
    #print('H: '+str(h))
    #print('E: '+str(e))
    #print('X and Y: ' + str(xy))
    return c,h,e,xy

def countStructures8(sequence):
    c=0
    h=0
    e=0
    t=0
    s=0
    b=0
    ii=0
    g=0
    none=0
    xy=0
    for i in range(len(sequence)):
        if(sequence[i]=='C'):
            c+=1
        elif(sequence[i]=='H'):
            h+=1
	elif(sequence[i]=='I'):
            ii+=1
        elif(sequence[i]=='E'):
            e+=1
        elif (sequence[i] == 'G'):
            g += 1
        elif (sequence[i] == 'T'):
            t += 1
        elif (sequence[i] == 'S'):
            s += 1
        elif (sequence[i] == 'B'):
            b += 1
        elif (sequence[i] == '-'):
            none += 1
        elif (sequence[i] == 'Y' or sequence[i]=='X'):
            xy += 1
        else:
            #raise ValueError('Unknown structure', sequence[i])
	    print('unknown found:', sequence[i])
	    return 0,0,0,0,0,0,0,0,0,0
    '''
    print('######')
    print('C: '+str(c))
    print('H: '+str(h))
    print('E: '+str(e))
    print('G: ' + str(g))
    print('T: ' + str(t))
    print('S: ' + str(s))
    print('B: ' + str(b))
    print('-: ' + str(none))
    print('xy: '+str(xy))
    '''
    return c,h,e,g,t,s,b,none, xy,ii

dict=getInput(proteins_path, 3)
print('dict one read!', len(dict))

c=0
h=0
e=0
xy=0

for i in range(len(dict)):
    tmp_c,tmp_h,tmp_e, tmp_xy=countStructures3(dict[dict.keys()[i]][1])
    c+=tmp_c
    e+=tmp_e
    h+=tmp_h
    xy+=tmp_xy
print('C: ',c,' , H: ',h,' , E: ',e,' ,XY: ',xy)

dict=getInput(proteins_path, 8)
print('dict two read!', len(dict))

c=0
h=0
e=0
xy=0
g=0
t=0
b=0
s=0
ii=0
none=0

for i in range(len(dict)):
    print(dict.keys()[i])
    tmp_c,tmp_h,tmp_e, tmp_g,tmp_t,tmp_s,tmp_b, tmp_none, tmp_xy, tmp_ii=countStructures8(dict[dict.keys()[i]][1])
    c+=tmp_c
    e+=tmp_e
    h+=tmp_h
    none+=tmp_none
    ii+=tmp_ii
    s += tmp_s
    b += tmp_b
    t += tmp_t
    g += tmp_g
    xy+=tmp_xy
print('C: ',c,'I:',ii,' , H: ',h,' , E: ',e,' T: ',t,' G: ',g,'B: ',b,' S: ',s,' .: ',none,' ,XY: ',xy)
