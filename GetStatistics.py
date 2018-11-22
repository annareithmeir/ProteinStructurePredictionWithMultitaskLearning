import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os

proteins_path='../mheinzinger/contact_prediction_v2/targets/dssp'
seq_path='../mheinzinger/contact_prediction_v2/sequences'
statistics_dir='stats'

attributes={}

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

def countStructures3(dict):
    c = 0
    h = 0
    e = 0
    xy = 0
    for sample in dict.keys():
        sequence=dict[sample][1]
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

    counts=np.array([c,h,e,xy])
    np.save(statistics_dir+'/countClasses_3',counts)

def countStructures8(dict):
    h = 0
    e = 0
    t = 0
    s = 0
    b = 0
    ii = 0
    g = 0
    none = 0
    xy = 0
    for sample in dict.keys():
        sequence = dict[sample][1]
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
                print('unknown found:', sequence[i])
                return 0,0,0,0,0,0,0,0,0
    counts=np.array([h,e,ii,s,t,g,b,none,xy])
    np.save(statistics_dir + '/countClasses_8', counts)

def countStructureChains3_dict(dict):
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

    C=[]
    H=[]
    E=[]
    counts_all = pd.DataFrame(np.zeros((0, 3), dtype=int),
                              columns=['C', 'H', 'E'],
                              index=[range(0)])

    for sample in dict.keys():
        counts_tmp, C_tmp,H_tmp,E_tmp = countStructureChains3(dict[sample][1])
        C.extend(C_tmp)
        E.extend(E_tmp)
        H.extend(H_tmp)
        counts_all = counts_all.append(counts_tmp)

    counts=counts_all.groupby(counts_all.index).sum()
    counts.to_pickle(statistics_dir+'/countChains3')
    tmp={'avg3':[np.average(C), np.average(H), np.average(E)]}
    attributes.update(tmp)

def countStructureChains8_dict(dict):
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

        matrix = np.zeros((max(len(np.bincount(H)), len(np.bincount(E)), len(np.bincount(I)),
                               len(np.bincount(S)), len(np.bincount(T)), len(np.bincount(G)), len(np.bincount(B)),
                               len(np.bincount(none))), 8), dtype=int)
        counts = pd.DataFrame(matrix,
                              columns=['H', 'E', 'I', 'S', 'T', 'G', 'B', '-'],
                              index=[range(
                                  max(len(np.bincount(H)), len(np.bincount(E)), len(np.bincount(I)),
                                      len(np.bincount(S)), len(np.bincount(T)), len(np.bincount(G)),
                                      len(np.bincount(B)),
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
        return counts, H, E, I, S, T, G, B, none

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
    for sample in dict.keys():
        counts_tmp, H_tmp, E_tmp, I_tmp, S_tmp, T_tmp, G_tmp, B_tmp, none_tmp = countStructureChains8(dict[sample][1])
        E.extend(E_tmp)
        H.extend(H_tmp)
        I.extend(I_tmp)
        S.extend(S_tmp)
        T.extend(T_tmp)
        B.extend(B_tmp)
        G.extend(G_tmp)
        none.extend(none_tmp)
        counts_all = counts_all.append(counts_tmp)

    counts=counts_all.groupby(counts_all.index).sum()
    counts.to_pickle(statistics_dir+'/countChains8')

    tmp={'avg8':[np.average(H), np.average(E),np.average(I),np.average(S),np.average(T),np.average(G),np.average(B),np.average(none)]}
    attributes.update(tmp)

def countAminoAcidsPerStruct_dict(dict3, dict8):
    def countAminoAcidsPerStruct8(seq, struct):
        matrix = np.zeros((9, 21), dtype=int)
        aa_counts = pd.DataFrame(matrix,  # # 0:H, 1:E, 2:I, 3:S, 4:T, 5:G, 6:B, 7:-
                                 columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
                                          'S',
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

    def countAminoAcidsPerStruct3(seq, struct):
        matrix = np.zeros((4, 21), dtype=int)
        aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                                 columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
                                          'S',
                                          'T',
                                          'W', 'Y', 'V', 'Else'],
                                 index=[np.arange(4)])
        for i in range(len(seq)):
            if (struct[i] == 'C'):
                if (seq[i] in aa_counts.keys()):
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

    matrix = np.zeros((4, 21), dtype=int)
    aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                             columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                      'T',
                                      'W', 'Y', 'V', 'Else'],
                             index=[np.arange(4)])

    for sample in dict3.keys():
        seq = dict3[sample][0]
        struct = dict3[sample][1]
        aa_counts_tmp=countAminoAcidsPerStruct3(seq,struct)
        aa_counts=aa_counts.append(aa_counts_tmp)

    counts = np.array([aa_counts.groupby(aa_counts.index).sum()])
    np.save(statistics_dir + '/countAAs3', counts)

    matrix = np.zeros((9, 21), dtype=int)
    aa_counts = pd.DataFrame(matrix,  # 0:C,1:H,2:E,3:XY
                             columns=['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
                                      'S',
                                      'T',
                                      'W', 'Y', 'V', 'Else'],
                             index=[np.arange(9)])

    for sample in dict8.keys():
        seq = dict8[sample][0]
        struct = dict8[sample][1]
        aa_counts_tmp = countAminoAcidsPerStruct8(seq, struct)
        aa_counts=aa_counts.append(aa_counts_tmp)

    counts=np.array([aa_counts.groupby(aa_counts.index).sum()])
    np.save(statistics_dir+'/countAAs8',counts)


def countDistanceBetweenBetaSheets_dict(dict):
    def countDistanceBetweenBetaSheets(sequence):
        dist = []
        i = 0
        tmp = 0
        while i < len(sequence):
            if ((sequence[i] == 'B' or sequence[i] == 'E') and tmp > 0):
                dist.append(tmp)
                tmp = 0
            else:
                tmp += 1
            i += 1
        if (len(dist) > 0):
            del dist[0]

        # print('beta dist:', dist)
        return dist

    dist=[]
    for sample in dict.keys():
        seq=dict[sample][1]
        dist_tmp=countDistanceBetweenBetaSheets(seq)
        dist=dist+dist_tmp
    return np.average(dist)


dict3, lengths3 = getInput(proteins_path, 3)
dict8, lengths8 = getInput(proteins_path, 8)

countStructures3(dict3)
countStructures8(dict8)
countStructureChains3_dict(dict3)
countStructureChains8_dict(dict8)

np.save(statistics_dir+'/lengths3',np.array(lengths3))
print(len(attributes))
np.save(statistics_dir+'/stats_dict', attributes)