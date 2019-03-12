networks/
  Networks.py -- All neural network models are here
  {NETWORK}_{TASKTYPE}_final.py -- network for tasktype

log/
  Structure/
    {NETWORK}_{INPUTTYPE}_structure -- Log for network, structure pred and specific inputtype
  MultiTask/
    {NETWORK}_{INPUTTYPE}_multitask -- Log for network, multitask pred and specific inputtype

pssm/
  --Outputs of the PSSM algorithm for the MSAs as numpy arrays, concatenated to ProtVec matrices

stats/
  --Data analysis results


preprocessing/
  --Each Protein folder contains inputs/targets as numpy arrays




NETWORKS: {CNN, LSTM, biLSTM, DenseCNN, Hybrid}
TASKTYPE: {Structure, Multitask}
INPUTTYPE:{1hot, protvec, protvecevolutionary, protvec+scoringmatrix, protvec+allpssm}