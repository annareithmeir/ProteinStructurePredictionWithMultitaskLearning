import utilities


'''
print('CNN 1hot')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/CNN_1hot_3_multitask')
print('CNN protvec')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/CNN_protvec_3_multitask')
print('CNN pssm')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/CNN_pssm_3_multitask')
print('CNN pssm+protvec')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/CNN_protvec+scoringmatrix_3_multitask')

print('LSTM')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/LSTM_protvec+scoringmatrix_3_multitask')
print('biLSTM')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/biLSTM_protvec+scoringmatrix_3_multitask')
print('DenseCNN')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/DenseCNN_protvec+scoringmatrix_3_multitask')
print('Hybrid')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/Hybrid_protvec+scoringmatrix_3_multitask')
print('DenseHybrid')
utilities.get_other_scores_fromfile('/home/areithmeier/log/struct/3/DenseHybrid_protvec+scoringmatrix_3_multitask')

print('multi2')
utilities.get_other_scores_fromfile('/home/areithmeier/log/multi2/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
print('multi3')
utilities.get_other_scores_fromfile('/home/areithmeier/log/multi3/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
print('multi4')
utilities.get_other_scores_fromfile('/home/areithmeier/log/multi4/3/DenseHybrid_protvec+scoringmatrix_3_multitask')
'''
print('DSSP8')
utilities.get_other_scores_fromfile('/home/areithmeier/log/multi4/8/DenseHybrid_protvec+scoringmatrix_8_multitask')
utilities.plot_confmat('/home/areithmeier/log/multi4/8/DenseHybrid_protvec+scoringmatrix_8_multitask')
utilities.plot_bootstrapping('/home/areithmeier/log/struct/3/biLSTM_protvec+scoringmatrix_3_multitarget')