#!/bin/sh
#$ -S /bin/sh

#
# This file trains and tests all networks with the configurations for the thesis
#

if [ -f compareNumberOfParameters.txt ]; then
rm compareNumberOfParameters.txt
fi

NEPOCHS=100
DSSP_MODE=3
MAP=True
MULTI_MODE=struct

#####################################################

#
# 1. CNN: 1hot vs. ProtVec?
#

python networks/CNN.py 'protvec' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/CNN.py '1hot' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}

#
# 2. CNN: PSSM vs. ProtVec+PSSM vs. ProtVecEvolutionary?
#

python networks/CNN.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/CNN.py 'pssm' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/CNN.py 'protvecevolutionary' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}

#
# 3. Which network performs best on protvec+scoringmatrix?
#

python networks/LSTM.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/biLSTM.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/DenseCNN.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/Hybrid.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}
python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}


#
# 3. Which multimode is best for DenseHybrid and protvec+scoringmatrix?
#

MULTI_MODE=multi2
python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}

MULTI_MODE=multi3
python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}

MULTI_MODE=multi4
python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} ${DSSP_MODE} ${MAP} ${MULTI_MODE}


#
# 4. DSSP8 on DenseHybrid
#

python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} 8 False multi2

#
# 5. DSSP8_mapped on DenseHybrid
#

python networks/DenseHybrid.py 'protvec+scoringmatrix' ${NEPOCHS} 8 True multi2

#
# 6. protvecevolutionary for DenseHybrid and DenseCNN
#

python networks/DenseHybrid.py 'protvecevolutionary' 100 3 True struct
python networks/DenseHybrid.py 'protvecevolutionary' 100 3 True multi2
python networks/DenseHybrid.py '1hot' 100 3 True multi2
python networks/DenseHybrid.py 'protvec' 100 3 True multi2
python networks/DenseHybrid.py 'pssm' 100 3 True multi2

##################################################





