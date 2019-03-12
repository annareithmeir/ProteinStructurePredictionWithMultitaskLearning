#!/bin/sh
#$ -S /bin/sh
#$ -t 1-4214

#
# This file collects all PSSM representations in one folder
#


cd /mnt/home/reithmeir/datapreprocessing
dirs=($(ls -d *))
PROT=${dirs[$SGE_TASK_ID-1]}
cd ${PROT}

if [ -f /mnt/home/reithmeir/datapreprocessing/${PROT}/${PROT}_msa.fasta ]; then
    mkdir /mnt/home/reithmeir/pssms/${PROT}
    cp /mnt/home/reithmeir/datapreprocessing/${PROT}/protvec+scoringmatrix.npy /mnt/home/reithmeir/datapreprocessing/${PROT}/protvec+information.npy /mnt/home/reithmeir/datapreprocessing/${PROT}/protvec+gapweights.npy /home/usr/rapid/ /mnt/home/reithmeir/pssms/${PROT}/
fi