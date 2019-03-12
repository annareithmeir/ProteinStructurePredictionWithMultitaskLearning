#!/bin/sh
#$ -S /bin/sh

NEPOCHS=10
python networks/CNN_Structure_final.py 'protvecevolutionary' ${NEPOCHS}
python networks/CNN_Structure_final.py 'protvec+scoringmatrix' ${NEPOCHS}

python networks/DenseCNN_Structure_final.py 'protvecevolutionary' ${NEPOCHS}
python networks/DenseCNN_Structure_final.py 'protvec+scoringmatrix' ${NEPOCHS}

python networks/CNN_LSTM_Structure_final.py 'protvecevolutionary' ${NEPOCHS}
python networks/CNN_LSTM_Structure_final.py 'protvec+scoringmatrix' ${NEPOCHS}

python networks/LSTM_Structure_final.py 'protvecevolutionary' ${NEPOCHS}
python networks/LSTM_Structure_final.py 'protvec+scoringmatrix' ${NEPOCHS}

python networks/biLSTM_Structure_final.py 'protvecevolutionary' ${NEPOCHS}
python networks/biLSTM_Structure_final.py 'protvec+scoringmatrix' ${NEPOCHS}

python compareResults.py