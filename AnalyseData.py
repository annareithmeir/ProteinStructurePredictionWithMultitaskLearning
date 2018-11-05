import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os
from math import ceil

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
    lengths=[]
    for prot in os.listdir(folder):
        res=readResidues(prot)
        lengths.append(len(res))
        if(classification_mode==3):
            str=readStructures3(prot)
        elif(classification_mode==8):
            str=readStructures8(prot)
        else:
            raise ValueError('[ERROR] Either 3 or 8 classes')
	if(str!=None and res!=None):
            tmp = {prot: (res, str)}
            dict.update(tmp)
    return dict, lengths

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
        if(sequence[i]=='H'):
            h+=1
        elif (sequence[i] == 'I'):
            ii += 1
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
	    return 0,0,0,0,0,0,0,0,0
    return h,e,g,t,s,b,none, xy,ii

def countStructureChains3(sequence):
    C = []
    H = []
    E = []
    i = 0

    while i < len(sequence):
        tmp = sequence[i]
        cnt = 1

        if (i < len(sequence) - 1):
            while (sequence[i + 1] == tmp):
                cnt += 1
                i = i + 1
                if (i == len(sequence) - 1):
                    break
        if (tmp == 'C'):
            C.insert(0, cnt)
        elif (tmp == 'H'):
            H.insert(0, cnt)
        elif (tmp == 'E'):
            E.insert(0, cnt)
        i = i + 1

    matrix = np.zeros((max(len(np.bincount(C)), len(np.bincount(H)), len(np.bincount(E))), 3), dtype=int)
    counts = pd.DataFrame(matrix,
                          columns=['C', 'H', 'E'],
                          index=[range(max(len(np.bincount(C)), len(np.bincount(H)), len(np.bincount(E))))])

    for i in range(len(np.bincount(C))):
        counts['C'][i] = np.bincount(C)[i]
    for i in range(len(np.bincount(H))):
        counts['H'][i] = np.bincount(H)[i]
    for i in range(len(np.bincount(E))):
        counts['E'][i] = np.bincount(E)[i]

    # plot3combined(counts)
    return counts, C, H, E

def countStructureChains8(sequence):
    H = []
    E = []
    I = []
    T = []
    S = []
    B = []
    G = []
    none = []
    i = 0

    while i < len(sequence):
        tmp = sequence[i]
        cnt = 1

        if (i < len(sequence) - 1):
            while (sequence[i + 1] == tmp):
                cnt += 1
                i = i + 1
                if (i == len(sequence) - 1):
                    break
        if (tmp == 'H'):
            H.insert(0, cnt)
        elif (tmp == 'S'):
            S.insert(0, cnt)
        elif (tmp == 'B'):
            B.insert(0, cnt)
        elif (tmp == 'G'):
            G.insert(0, cnt)
        elif (tmp == 'T'):
            T.insert(0, cnt)
        elif (tmp == 'I'):
            I.insert(0, cnt)
        elif (tmp == '-'):
            none.insert(0, cnt)
        elif (tmp == 'E'):
            E.insert(0, cnt)
        i = i + 1

    matrix = np.zeros((max( len(np.bincount(H)), len(np.bincount(E)), len(np.bincount(I)),
                           len(np.bincount(S)), len(np.bincount(T)), len(np.bincount(G)), len(np.bincount(B)),
                           len(np.bincount(none))), 8), dtype=int)
    counts = pd.DataFrame(matrix,
                          columns=['H', 'E', 'I', 'S', 'T', 'G', 'B', '-'],
                          index=[range(
                              max(len(np.bincount(H)), len(np.bincount(E)), len(np.bincount(I)),
                                  len(np.bincount(S)), len(np.bincount(T)), len(np.bincount(G)), len(np.bincount(B)),
                                  len(np.bincount(none))))])

    for i in range(len(np.bincount(H))):
        counts['H'][i] = np.bincount(H)[i]
    for i in range(len(np.bincount(E))):
        counts['E'][i] = np.bincount(E)[i]
    for i in range(len(np.bincount(I))):
        counts['I'][i] = np.bincount(I)[i]
    for i in range(len(np.bincount(S))):
        counts['S'][i] = np.bincount(S)[i]
    for i in range(len(np.bincount(T))):
        counts['T'][i] = np.bincount(T)[i]
    for i in range(len(np.bincount(G))):
        counts['G'][i] = np.bincount(G)[i]
    for i in range(len(np.bincount(none))):
        counts['-'][i] = np.bincount(none)[i]
    for i in range(len(np.bincount(B))):
        counts['B'][i] = np.bincount(B)[i]

    # plot8combined(counts)
    return counts, H,E,I,S,T,B,G,none

def countStructureChains3_dict(dict):
    C=[]
    H=[]
    E=[]
    counts_all = pd.DataFrame(np.zeros((0, 3), dtype=int),
                              columns=['C', 'H', 'E'],
                              index=[range(0)])

    f = open('structureChains.txt', 'w')
    for i in range(len(dict)):
        counts_tmp, C_tmp,H_tmp,E_tmp = countStructureChains3(dict[dict.keys()[i]][1])
        f.write(dict.keys()[i])
        f.write(counts_tmp.to_string())
        f.write(' ')
        C.extend(C_tmp)
        E.extend(E_tmp)
        H.extend(H_tmp)
        counts_all = counts_all.append(counts_tmp)
    f.close()
    return (counts_all.groupby(counts_all.index).sum()), np.average(C), np.average(H), np.average(E)

def countStructureChains8_dict(dict):
    H = []
    E = []
    I = []
    T = []
    S = []
    B = []
    G = []
    none = []
    counts_all = pd.DataFrame(np.zeros((0, 8), dtype=int),
                              columns=[ 'H', 'E', 'I', 'S', 'T', 'G', 'B', '-'],
                              index=[range(0)])
    f = open('structureChains8.txt', 'w')
    for i in range(len(dict)):
        counts_tmp, H_tmp, E_tmp, I_tmp, S_tmp, T_tmp, B_tmp, G_tmp, none_tmp = countStructureChains8(dict[dict.keys()[i]][1])
        f.write(dict.keys()[i])
        f.write(counts_tmp.to_string())
        f.write(' ')
        E.extend(E_tmp)
        H.extend(H_tmp)
        I.extend(I_tmp)
        S.extend(S_tmp)
        T.extend(T_tmp)
        B.extend(B_tmp)
        G.extend(G_tmp)
        none.extend(none_tmp)
        counts_all = counts_all.append(counts_tmp)
    f.close()
    return (counts_all.groupby(counts_all.index).sum()), np.average(H), np.average(E), np.average(I), np.average(S), np.average(T), np.average(B), np.average(G), np.average(none)

def plot3(counts, avgs):
    plt.figure()
    plt.suptitle('Occurences of chain lengths')
    plt.title(r'Number of how long the structure chains are')
    plt.ylabel('Occurences')
    cols = ['green', 'cyan', 'orange', 'blue', 'red', 'yellow', 'pink', 'brown']

    for i in range(counts.shape[1]):
        tmp = counts.keys()[i]
        plt.subplot(1, counts.shape[1], i + 1)
        plt.axvline(avgs[i], color='grey', linestyle='dashed',
                    linewidth=1)
        plt.xlabel('Chain length ' + tmp + ' \n(max:' + str(np.max(counts[tmp])) + ' min:' + str(
            np.min(counts[tmp])) + ' total:' + str(np.sum(counts[tmp])) + ')')
        plt.ylim([0, np.max(counts.values)])
        plt.bar(range(counts.shape[0]), counts[tmp], label='C', color=cols[i])

    plt.savefig('countChains3_each.pdf')

def plot8(counts, avgs):
    plt.figure()
    plt.suptitle('Occurences of chain lengths')
    plt.title(r'Number of how long the structure chains are')
    plt.ylabel('Occurences')
    cols = ['green', 'cyan', 'orange', 'blue', 'red', 'yellow', 'pink', 'brown', 'black']

    for i in range(counts.shape[1]):
        tmp = counts.keys()[i]
        plt.subplot(1, counts.shape[1], i+1)
        plt.axvline(avgs[i], color='grey', linestyle='dashed', linewidth=1)
        plt.xlabel('Chain length '+tmp+' \n(max:'+str(np.max(counts[tmp]))+' min:'+str(np.min(counts[tmp]))+' total:'+str(np.sum(counts[tmp]))+')')
        plt.ylim([0, np.max(counts.values)])
        plt.bar(range(counts.shape[0]),counts[tmp],  label='C', color=cols[i])

        plt.savefig('countChains8_each.pdf')

def countAminoAcidsPerStruct3(seq, struct):
    matrix = np.zeros((4, 21), dtype=int)
    aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                             columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                      'T',
                                      'W', 'Y', 'V', 'Else'],
                             index=[np.arange(4)])
    for i in range(len(seq)):
        if (struct[i] == 'C'):
            if(seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][0] += 1
            else:
                aa_counts['Else'][0] += 1
        elif (struct[i] == 'H'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][1] += 1
            else:
                aa_counts['Else'][1] += 1
        elif (struct[i] == 'E'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][2] += 1
            else:
                aa_counts['Else'][2] += 1
        else:
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][3] += 1
            else:
                aa_counts['Else'][3] += 1
    return aa_counts

def countAminoAcidsPerStruct8(seq, struct):
    matrix = np.zeros((9, 21), dtype=int)
    aa_counts = pd.DataFrame(matrix,  #  # 0:H, 1:E, 2:I, 3:S, 4:T, 5:G, 6:B, 7:-
                             columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                      'T',
                                      'W', 'Y', 'V', 'Else'],
                             index=[np.arange(9)])
    for i in range(len(seq)):
        if (struct[i] == 'H'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][0] += 1
            else:
                aa_counts['Else'][0] += 1
        elif (struct[i] == 'E'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][1] += 1
            else:
                aa_counts['Else'][1] += 1
        elif (struct[i] == 'I'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][2] += 1
            else:
                aa_counts['Else'][2] += 1
        elif (struct[i] == 'S'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][3] += 1
            else:
                aa_counts['Else'][3] += 1
        elif (struct[i] == 'T'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][4] += 1
            else:
                aa_counts['Else'][4] += 1
        elif (struct[i] == 'G'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][5] += 1
            else:
                aa_counts['Else'][5] += 1
        elif (struct[i] == 'B'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][6] += 1
            else:
                aa_counts['Else'][6] += 1
        elif (struct[i] == '-'):
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][7] += 1
            else:
                aa_counts['Else'][7] += 1
        else:
            if (seq[i] in aa_counts.keys()):
                aa_counts[seq[i]][8] += 1
            else:
                aa_counts['Else'][8] += 1
    return aa_counts

def countAminoAcidsPerStruct_dict(dict, classification_mode):
    if(classification_mode==3):
        matrix = np.zeros((4, 21), dtype=int)
        aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                                 columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                          'T',
                                          'W', 'Y', 'V', 'Else'],
                                 index=[np.arange(4)])
    elif(classification_mode==8):
        matrix = np.zeros((9, 21), dtype=int)
        aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                                 columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
                                          'S',
                                          'T',
                                          'W', 'Y', 'V', 'Else'],
                                 index=[np.arange(9)])
    else:
        raise ValueError('either 3 or 8 classes')

    for i in range(len(dict)):
        seq = dict[dict.keys()[i]][0]
        struct = dict[dict.keys()[i]][1]
        if(classification_mode==3):
            aa_counts_tmp=countAminoAcidsPerStruct3(seq,struct)
        else:
            aa_counts_tmp = countAminoAcidsPerStruct8(seq, struct)
        aa_counts=aa_counts.append(aa_counts_tmp)

    return aa_counts.groupby(aa_counts.index).sum()

def countDistanceBetweenBetaSheets(sequence):
    dist=[]
    i=0
    tmp=0
    while i<len(sequence):
        if(sequence[i]=='B' or sequence[i]=='E'):
            dist.append(tmp)
            tmp=0
        else:
            tmp+=1
        i+=1
    if(len(dist)>0):
        del dist[0]
    return dist

def countDistanceBetweenBetaSheets_dict(dict):
    dist=[]
    for i in range(len(dict)):
        seq=dict[dict.keys()[i]][1]
        print(seq)
        dist_tmp=countDistanceBetweenBetaSheets(seq)
        dist=dist+dist_tmp

    return dist

## SCRIPT ##
def analyseData():
    f = open('results.txt', 'w')

    dict, lengths = getInput(proteins_path, 3)
    print('dict one read!', len(dict), ', average sequence length: ',np.average(lengths))
    c = 0
    h = 0
    e = 0
    xy = 0
    for i in range(len(dict)):
        tmp_c, tmp_h, tmp_e, tmp_xy = countStructures3(dict[dict.keys()[i]][1])
        c += tmp_c
        e += tmp_e
        h += tmp_h
        xy += tmp_xy
    f.write('Average sequence length: ')
    f.write(np.average(lengths))
    f.write('Classification of targets (3 classes)\n')
    f.write(str('C: ' + str(c) + ' , H: ' + str(h) + ' , E: ' + str(e) + ' ,XY: ' + str(xy) + '\n'))
    chains, C_avg, H_avg, E_avg = countStructureChains3_dict(dict)
    f.write('Counting structure chains (3 classes)\n')
    f.write(chains.to_string())
    f.write('\n')
    f.write(str('Average C length: ' + str(C_avg) + '\n'))
    f.write(str('Average H length: ' + str(H_avg) + '\n'))
    f.write(str('Average E length: ' + str(E_avg) + '\n'))
    f.write('\n')
    aa=countAminoAcidsPerStruct_dict(dict,3)
    f.write('Counts of AAs per structure class 0=C, 1=H, 2=E, 3=XY\n')
    f.write(aa.to_string())
    f.write('\n')
    beta = countDistanceBetweenBetaSheets_dict(dict)
    avg_beta=sum(beta)/float(len(beta))
    f.write('Average distance between beta-sheets:')
    f.write(str(avg_beta))
    f.write('\n')


    dict = getInput(proteins_path, 8)
    print('dict two read!', len(dict))
    h = 0
    e = 0
    xy = 0
    g = 0
    t = 0
    b = 0
    s = 0
    ii = 0
    none = 0
    for i in range(len(dict)):
        tmp_h, tmp_e, tmp_g, tmp_t, tmp_s, tmp_b, tmp_none, tmp_xy, tmp_ii = countStructures8(
            dict[dict.keys()[i]][1])
        e += tmp_e
        h += tmp_h
        none += tmp_none
        ii += tmp_ii
        s += tmp_s
        b += tmp_b
        t += tmp_t
        g += tmp_g
        xy += tmp_xy

    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write('classification of targets (8 classes)\n')
    f.write(str( 'I: ' + str(ii) + ' , H: ' + str(h) + ' , E: ' + str(e) + ' T: ' + str(t) + ' G: ' + str(g) + 'B: ' + str(b) + ' S: ' + str(s) + ' -: ' + str(none) + ' ,XY: ' + str(xy) + '\n'))
    chains,  H_avg, E_avg, I_avg, S_avg, T_avg, B_avg, G_avg, none_avg = countStructureChains8_dict(dict)
    f.write('Counting structure chains (8 classes)\n')
    f.write(chains.to_string())
    f.write('\n')
    f.write(str('Average H length: ' + str(H_avg) + '\n'))
    f.write(str('Average E length: ' + str(E_avg) + '\n'))
    f.write(str('Average I length: ' + str(I_avg) + '\n'))
    f.write(str('Average S length: ' + str(S_avg) + '\n'))
    f.write(str('Average T length: ' + str(T_avg) + '\n'))
    f.write(str('Average B length: ' + str(B_avg) + '\n'))
    f.write(str('Average G length: ' + str(G_avg) + '\n'))
    f.write(str('Average none length: ' + str(none_avg) + '\n'))
    aa = countAminoAcidsPerStruct_dict(dict, 8)
    f.write('Counts of AAs per structure class 0=H, 1=E, 2=I, 3=S, 4=T, 5=G, 6=B, 7=-, 8=XY\n')
    f.write(aa.to_string())
    f.write('\n')
    beta = countDistanceBetweenBetaSheets_dict(dict)
    avg_beta = sum(beta) / float(len(beta))
    f.write('Average distance between beta-sheets:')
    f.write(str(avg_beta))
    f.write('\n')
    f.close()

analyseData()