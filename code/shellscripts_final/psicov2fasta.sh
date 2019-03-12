#!/bin/sh
#$ -S /bin/sh
#$ -t 1-4214

#
# This file first creates a fasta file for each MSA and then calls the PSIBLAST algorithm on that to
# create the PSSM, which is then parsed
#


cd /mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments
dirs=($(ls -d *))
PROT=${dirs[$SGE_TASK_ID-1]}
IN=$PROT

if [ -f /mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments/${PROT}/${PROT}.psicov ]; then
    /mnt/project/multitarget/files/psicov2fasta.py $IN
    if [ -f /mnt/project/multitarget/fasta_fix/${PROT}_msa.fasta ]; then
        psiblast -subject /mnt/project/multitarget/fasta_fix/${PROT}_msa.fasta -in_msa /mnt/project/multitarget/fasta_fix/${PROT}_msa.fasta -out_ascii_pssm /mnt/project/multitarget/PSSM_fix/${PROT}_pssm.txt
        /mnt/project/multitarget/files/parsePSSM.py $PROT
    fi
fi
