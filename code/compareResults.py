import utilities
import sys

#
# This file handles all plotting of the confusion matrices and accuracy distributions
#

# Compare CNN for all input types without multitarget (for Q3)
utilities.plotComparisonForNetworkType(str('/home/areithmeier/log/struct/3/'), 'CNN')
utilities.plot_confmat('/home/areithmeier/log/struct/3/CNN_protvec+scoringmatrix_3_multitask')
utilities.plot_confmat('/home/areithmeier/log/struct/3/CNN_protvec_3_multitask')
utilities.plot_confmat('/home/areithmeier/log/struct/3/CNN_pssm_3_multitask')
utilities.plot_confmat('/home/areithmeier/log/struct/3/CNN_1hot_3_multitask')
utilities.plot_confmat('/home/areithmeier/log/struct/3/CNN_protvecevolutionary_3_multitask')

utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/CNN_protvecevolutionary_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/CNN_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/CNN_protvec_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/CNN_1hot_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/CNN_pssm_3_multitask')

# Compare how different nets perform for protvec+scoringmatrix
utilities.plotComparisonForInputType(str('log/struct/3/'), 'protvec+scoringmatrix', 'struct')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/biLSTM_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/LSTM_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/DenseCNN_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/Hybrid_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/struct/3/DenseHybrid_protvec+scoringmatrix_3_multitask')

#compare multitarget mode for DenseHybrid
utilities.plotComparisonForMultitargetType('log/','DenseHybrid', ['protvec+scoringmatrix'], ['3'])
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/multi3/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask')

utilities.plot_confmat('/home/areithmeier/log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask')

utilities.plotComparisonForNetworkType(str('/home/areithmeier/log/multi2/3/'), 'DenseHybrid')
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/3/DenseHybrid_protvec_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/3/DenseHybrid_1hot_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/3/DenseHybrid_pssm_3_multitask')
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/3/DenseHybrid_protvecevolutionary_3_multitask')

#DSSP8
utilities.plot_collected_confmats8('/home/areithmeier/log/multi2/8/DenseHybrid_protvec+scoringmatrix_8_multitask')
utilities.plot_confmat('/home/areithmeier/log/multi2/8/DenseHybrid_protvec+scoringmatrix_8_multitask')
#utilities.plotComparisonForMultitargetType('log/','DenseHybrid', ['protvec+scoringmatrix'], ['8'])

#DSSP8_MAPPED
utilities.plot_collected_confmats('/home/areithmeier/log/multi2/8_mapped/DenseHybrid_protvec+scoringmatrix_8_multitask')
utilities.plot_confmat('/home/areithmeier/log/multi2/8_mapped/DenseHybrid_protvec+scoringmatrix_8_multitask')

#utilities.plotSolvAccFlex('/home/areithmeier/log/multi3/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
#utilities.plotSolvAccFlex('/home/areithmeier/log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
#utilities.plotSolvAccFlex('/home/areithmeier/log/multi2/3/DenseHybrid_protvec+scoringmatrix_3_multitask')




