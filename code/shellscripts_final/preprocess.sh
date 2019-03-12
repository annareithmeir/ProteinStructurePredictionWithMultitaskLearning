#
# This file pre-processes one protein at a time and produces the one-hot, protvecMSA-protvec
# representations.
#


#!/bin/sh
#$ -S /bin/sh
#$ -t 1-4214
cd /mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences
dirs=($(ls -d *))
PROT=${dirs[$SGE_TASK_ID-1]}
IN=/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences/"$PROT"
AL=/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/alignments/"$PROT"/"$PROT".psicov
PROT_VEC=/mnt/home/mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/protVecs/protVec_100d_3grams.csv
/mnt/home/reithmeir/preprocess_1hot.py $IN
/mnt/home/reithmeir/preprocess_structures.py $IN
/mnt/home/reithmeir/preprocess_protvec.py $IN
/mnt/home/reithmeir/preprocess_protvec_evolutionary.py $IN $AL

