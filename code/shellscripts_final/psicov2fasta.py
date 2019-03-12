#!/usr/bin/python3 -u
import os
import os.path
import sys

#
# This file generates a fasta file for a MSA
#

def insert_newlines(string, every=80): #Limits length of each line in fasta file to 80
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)

def write_fasta(protein): # Creates fasta file
    path='/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments/' + protein.upper() + '/' + protein.upper() + '.psicov'
    f = open(path, 'r')
    input = f.read().splitlines()
    f.close()

    length = 0
    for line in input:
        if (length == 0):
            length = len(line)
        else:
            if (len(line) != length):
                input.remove(line)

    out=open(save_path+'/'+protein+'_msa.fasta', 'w')
    for i in range(len(input)):
        msa=insert_newlines(input[i], 80)
        i=str(i)
        msa=msa.replace('-','X') #blast does not accept '-'as gap, instead use 'X'
        out.write('>INFO:')
        out.write(i)
        out.write('\n')
        out.write(msa)
        out.write('\n')
    out.close()


save_path='/mnt/project/multitarget/fasta_fix'
proteinName = sys.argv[1]

if(os.path.isfile('/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments/' + proteinName.upper() + '/' + proteinName.upper() + '.psicov')):
    if(not (os.path.exists('/mnt/project/multitarget/PSSM/'+proteinName.upper()+'_pssm.txt'))):
        write_fasta(proteinName)
else:
    print('no psicov file')