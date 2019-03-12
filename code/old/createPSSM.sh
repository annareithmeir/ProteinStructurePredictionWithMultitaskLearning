#!/bin/sh
#$ -S /bin/sh
#$ -t 1-4214
cd /mnt/home/reithmeir/datapreprocessing
dirs=($(ls -d *))
PROT=${dirs[$SGE_TASK_ID-1]}
cd ${PROT}

if [ -f /mnt/home/reithmeir/datapreprocessing/${PROT}/${PROT}_msa.fasta ]; then
    psiblast -subject /mnt/home/reithmeir/datapreprocessing/${PROT}/${PROT}_msa.fasta -in_msa /mnt/home/reithmeir/datapreprocessing/${PROT}/${PROT}_msa.fasta -out_ascii_pssm /mnt/home/reithmeir/datapreprocessing/${PROT}/${PROT}_pssm.txt
    /mnt/project/multitarget/files/ParsePSSM.py $PROT
fi