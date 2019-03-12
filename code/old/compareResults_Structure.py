import utilities




#Compare CNNfor all input types
utilities.plotComparisonForNetworkType('log/Structure', 'CNN', 'structure')
utilities.plotComparisonForNetworkType('log/Structure', 'LSTM', 'structure')
utilities.plotComparisonForNetworkType('log/Structure', 'biLSTM', 'structure')
utilities.plotComparisonForNetworkType('log/Structure', 'DenseCNN', 'structure')
utilities.plotComparisonForNetworkType('log/Structure', 'Hybrid', 'structure')


#Compare all networks for protvecevolutionary
utilities.plotComparisonForInputType('log/Structure', 'protvecevolutionary', 'structure')
utilities.plotComparisonForInputType('log/Structure', 'protvec+scoringmatrix', 'structure')