import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os

#
# Analyzes the predicted data, smaller version of getStatistics.py.
# Saves the statistics in npy files in the 'data analysis/predicted' folder
#

proteins_path='../mheinzinger/contact_prediction_v2/targets/dssp'
seq_path='../mheinzinger/contact_prediction_v2/sequences'
anna_path='dataset_preprocessed'
targets_path='/home/mheinzinger/contact_prediction_v2/targets'
statistics_dir='data analysis/predicted'

attributes={}

def readResidues(path):
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

def readStructures3(protein):
    path = proteins_path + '/'+protein.lower() + '/' + protein.lower() + '.3.consensus.dssp'
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
    path = proteins_path + '/'+ protein.lower() + '/' + protein.lower() + '.8.consensus.dssp'
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

def getInput(classification_mode): #proteins in a folder named root+proteins/
    dict={}
    lengths=[]
    i=0
    for prot in os.listdir(anna_path):
        i+=1
        res=readResidues(seq_path+'/'+prot.upper()+'.fasta.txt')
        lengths.append(len(res))
        if(classification_mode==3):
            str=readStructures3(prot.lower())
        elif(classification_mode==8):
            str=readStructures8(prot.lower())
        else:
            raise ValueError('[ERROR] Either 3 or 8 classes')

        solvAcc=np.memmap(targets_path + '/dssp/' + prot.lower() + '/' + prot.lower() + '.rel_asa.memmap',
                        dtype=np.float32, mode='r', shape=len(res))
        solvAcc=np.nan_to_num(solvAcc)

        flex=np.memmap(targets_path + '/bdb_bvals/' + prot.lower() + '.bdb.memmap', dtype=np.float32, mode='r', shape=len(res))
        flex=np.nan_to_num(flex)
        if(str!=None and res!=None and solvAcc!=None and flex!=None):
            tmp = {prot: (res, str, solvAcc, flex)}
            dict.update(tmp)
    return dict, lengths

def countStructures3(dict, f):
    c = 0
    h = 0
    e = 0
    xy = 0
    for sample in dict.keys():
        seq=dict[sample][0].cpu().numpy()[0]
        original_len=dict[sample][1].cpu().numpy()
        mask=dict[sample][2].cpu().numpy()[0]
        mask = mask[:original_len]
        xy_idx = np.where(mask == 0.0)
        seq[xy_idx] = 3
        sequence=''
        for i in seq:
            sequence+=str(i)
        sequence=sequence[:original_len]
        sequence=sequence.replace('0','C')
        sequence = sequence.replace('1', 'H')
        sequence = sequence.replace('2', 'E')
        sequence = sequence.replace('3', 'X')

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

    f.write('\n')
    f.write('###   DSSP3 Classes   ###\n')
    f.write(str('(C, H, E, XY): '+str(c)+' '+str(h)+' '+str(e)+' '+str(xy)+'\n'))
    f.write('\n')
    counts=np.array([c,h,e,xy])
    np.save(statistics_dir+'/countClasses_3',counts)
    tmp={'proportions3':counts}
    attributes.update(tmp)

def countStructures8(dict, f):
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
        seq = dict[sample][0].cpu().numpy()[0]
        original_len = dict[sample][1].cpu().numpy()
        mask = dict[sample][2].cpu().numpy()[0]
        mask = mask[:original_len]
        xy_idx = np.where(mask == 0.0)
        seq[xy_idx] = 8
        sequence = ''
        for i in seq:
            sequence += str(i)
        sequence = sequence[:original_len]
        sequence = sequence.replace('0', 'H')
        sequence = sequence.replace('1', 'E')
        sequence = sequence.replace('2', 'I')
        sequence = sequence.replace('3', 'S')
        sequence = sequence.replace('4', 'T')
        sequence = sequence.replace('5', 'G')
        sequence = sequence.replace('6', 'B')
        sequence = sequence.replace('7', '-')
        sequence = sequence.replace('8', 'X')
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
                return 0,0,0,0,0,0,0,0,0   #TODO: check if this is correct!

    f.write('\n')
    f.write('###   DSSP8 Classes   ###\n')
    f.write('(H, E, I, S, T, G, B, -, XY): '+str(h)+' '+str(e)+' '+str(i)+' '+str(s)+' '+str(t)+' '+str(g)+' '+str(b)+' '+str(none)+' '+str(xy)+'\n')
    f.write('\n')
    counts=np.array([h,e,ii,s,t,g,b,none,xy])
    np.save(statistics_dir + '/countClasses_8', counts)
    tmp={'proportions8':counts}
    attributes.update(tmp)

def countStructureChains3_dict(dict, f):
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
                    if (i == (len(sequence) - 1)):
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
                              index=[np.arange(max(len(np.bincount(C)), len(np.bincount(H)), len(np.bincount(E))))])

        for i in range(len(np.bincount(C))):
            counts['C'][i] = np.bincount(C)[i]
        for i in range(len(np.bincount(H))):
            counts['H'][i] = np.bincount(H)[i]
        for i in range(len(np.bincount(E))):
            counts['E'][i] = np.bincount(E)[i]

        return counts, C, H, E

    C=[]
    H=[]
    E=[]
    counts_all = pd.DataFrame(np.zeros((1, 3), dtype=int),
                              columns=['C', 'H', 'E'],
                              index=[np.arange(1)])

    for sample in dict.keys():
        seq = dict[sample][0].cpu().numpy()[0]
        original_len = dict[sample][1].cpu().numpy()
        mask = dict[sample][2].cpu().numpy()[0]
        mask = mask[:original_len]
        xy_idx = np.where(mask == 0.0)
        seq[xy_idx] = 3
        sequence = ''
        for i in seq:
            sequence += str(i)
        sequence = sequence[:original_len]
        sequence = sequence.replace('0', 'C')
        sequence = sequence.replace('1', 'H')
        sequence = sequence.replace('2', 'E')
        sequence = sequence.replace('3', 'X')
        counts_tmp, C_tmp,H_tmp,E_tmp = countStructureChains3(sequence)
        C.extend(C_tmp)
        E.extend(E_tmp)
        H.extend(H_tmp)
        counts_all = counts_all.append(counts_tmp)

    counts=counts_all.groupby(counts_all.index).sum()
    f.write('\n')
    f.write('###   DSSP3 Chains   ###\n')
    f.write('Average C, H, E: '+str(np.average(C))+' '+ str(np.average(H))+' '+ str(np.average(E)))
    f.write(counts.to_string())
    f.write('\n')
    counts.to_pickle(statistics_dir+'/countChains3')
    tmp={'avg3':[np.average(C), np.average(H), np.average(E)]}
    attributes.update(tmp)

def countStructureChains8_dict(dict, f):
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
                              index=[np.arange(
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
    counts_all = pd.DataFrame(np.zeros((1, 8), dtype=int),
                              columns=[ 'H', 'E', 'I', 'S', 'T', 'G', 'B', '-'],
                              index=[np.arange(1)])
    for sample in dict.keys():
        seq = dict[sample][0].cpu().numpy()[0]
        original_len = dict[sample][1].cpu().numpy()
        mask = dict[sample][2].cpu().numpy()[0]
        mask = mask[:original_len]
        xy_idx = np.where(mask == 0.0)
        seq[xy_idx] = 8
        sequence = ''
        for i in seq:
            sequence += str(i)
        sequence = sequence[:original_len]
        sequence = sequence.replace('0', 'H')
        sequence = sequence.replace('1', 'E')
        sequence = sequence.replace('2', 'I')
        sequence = sequence.replace('3', 'S')
        sequence = sequence.replace('4', 'T')
        sequence = sequence.replace('5', 'G')
        sequence = sequence.replace('6', 'B')
        sequence = sequence.replace('7', '-')
        sequence = sequence.replace('8', 'X')
        counts_tmp, H_tmp, E_tmp, I_tmp, S_tmp, T_tmp, G_tmp, B_tmp, none_tmp = countStructureChains8(sequence)
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
    f.write('\n')
    f.write('###   DSSP8 Chains   ###\n')
    f.write('Average H, E, I, S, T, G, B, -: '+ str(np.average(H))+' '+ str(np.average(E))+' '+ str(np.average(I))+' '+ str(np.average(S))+' '+
                                                str(np.average(T))+' '+str(np.average(G))+' '+str(np.average(B))+' '+str(np.average(none))+'\n')
    f.write(counts.to_string())
    f.write('\n')
    counts.to_pickle(statistics_dir+'/countChains8')

    tmp={'avg8':[np.average(H), np.average(E),np.average(I),np.average(S),np.average(T),np.average(G),np.average(B),np.average(none)]}
    attributes.update(tmp)

def countAminoAcidsPerStruct_dict(dict3, dict8, f):
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

    counts = aa_counts.groupby(aa_counts.index).sum()
    f.write('\n')
    f.write('###   TYPE OF AMINO ACIDS PER STRUCTURE CLASS DSSP3   ###\n')
    f.write(counts.to_string())
    f.write('\n')
    np.save(statistics_dir + '/countAAs3', np.array(counts))

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

    counts=aa_counts.groupby(aa_counts.index).sum()
    f.write('\n')
    f.write('###   TYPE OF AMINO ACIDS PER STRUCTURE CLASS DSSP8   ###\n')
    f.write(counts.to_string())
    f.write('\n')
    np.save(statistics_dir+'/countAAs8',np.array(counts))


def countDistanceBetweenBetaSheets_dict(dict, f):
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
        seq = dict[sample][0].cpu().numpy()[0]
        original_len = dict[sample][1].cpu().numpy()
        mask = dict[sample][2].cpu().numpy()[0]
        mask = mask[:original_len]
        xy_idx = np.where(mask == 0.0)
        seq[xy_idx] = 3
        sequence = ''
        for i in seq:
            sequence += str(i)
        sequence = sequence[:original_len]
        sequence = sequence.replace('0', 'C')
        sequence = sequence.replace('1', 'H')
        sequence = sequence.replace('2', 'E')
        seq = sequence.replace('3', 'X')

        dist_tmp=countDistanceBetweenBetaSheets(seq)
        dist=dist+dist_tmp

    f.write('\n')
    f.write('###   AVG DISTANCE BETWEEN BETA SHEETS   ###\n')
    f.write(str(np.average(dist)))
    f.write('\n')

def analyseFlex(dict):
    flex_avgs=[]
    three_bins=[0,0,0]
    for sample in dict.keys():
        flex=dict[sample][3]
        flex_avgs.append(np.average(flex))

        for f in flex:
            if(f<=-1):
                three_bins[0]+=1
            elif (f > -1 and f< 1):
                three_bins[1] += 1
            elif (f >= 1):
                three_bins[2] += 1

    flex_avgs=np.array(flex_avgs)
    three_bins=np.array(three_bins)
    np.save(statistics_dir+'/flex_avgs', flex_avgs)
    np.save(statistics_dir+'/flex_thirds', three_bins)
    tmp={'flex': three_bins}
    attributes.update(tmp)

def analyseSolvAcc(dict):
    solv_avgs = []
    four_bins = [0, 0, 0, 0]
    two_bins=[0,0]
    for sample in dict.keys():
        solv=dict[sample][2]
        solv_avgs.append(np.average(solv))

        for f in solv:
            if (f >= 0 and f < 0.25):
                four_bins[0] += 1
                two_bins[0] += 1
            elif (f >= 0.25 and f < 0.5):
                four_bins[1] += 1
                two_bins[0] += 1
            elif (f >= 0.5 and f < 0.75):
                four_bins[2] += 1
                two_bins[1] += 1
            elif (f >= 0.75 and f <= 1.0):
                four_bins[3] += 1
                two_bins[1] += 1

    solv_avgs = np.array(solv_avgs)
    four_bins = np.array(four_bins)
    two_bins = np.array(two_bins)
    np.save(statistics_dir + '/solvAcc_avgs', solv_avgs)
    np.save(statistics_dir + '/solvAcc_quarters', four_bins)
    np.save(statistics_dir + '/solvAcc_halfs', two_bins)

    tmp={'solvacc': two_bins}
    attributes.update(tmp)

# Analyze DSSP3 predictions
f = open('data analysis/predicted/statisticsPredicted.txt', 'w')

dict3 = np.load('/home/areithmeier/log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask/predictedData.npy').item()
countStructures3(dict3, f)

countStructureChains3_dict(dict3, f)
countDistanceBetweenBetaSheets_dict(dict3, f)

# Analyze DSSP8 predictions
f = open('data analysis/predicted/statisticsPredicted8.txt', 'w')
dict8 = np.load('/home/areithmeier/log/multi4/8/DenseHybrid_protvec+scoringmatrix_8_multitask/predictedData.npy').item()
countStructures8(dict8, f)

countStructureChains8_dict(dict8, f)
countDistanceBetweenBetaSheets_dict(dict8, f)

f.close()