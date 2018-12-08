import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    plt.switch_backend('agg')  # GPU is only available via SSH (no display)
import seaborn as sns

def create_logdir(log_path):
    # Try to create a new directory for saving log files
    if( not os.path.isdir(log_path)):
        os.mkdir(log_path)  # create directory for logging learning progress
        return log_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_bootstrapping(log_path, matrices):  # matrices=list of confusion matrices
    accs = []
    for i in range(len(matrices)):
        acc = matrices[i].trace() / float(matrices[i].sum())
        accs.append(round(acc, 2))

    plt.figure()
    sns.distplot(accs)
    plt.title('Accuracy distribution after last epoch ')
    print('standard deviation: ', np.std(np.array(accs), ddof=1))
    print('standard error: ', stats.sem(np.array(accs)))
    print('mean: ', np.average(np.array(accs)))
    plt.axvline(np.average(np.array(accs)), linestyle='dashed')
    plt.axvline(np.average(np.array(accs)) + np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.axvline(np.average(np.array(accs)) - np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.xlabel('Mean: ' + str(round(np.average(np.array(accs)), 3))+ '    Standard deviation: ' + str(round(np.std(np.array(accs), ddof=1), 3)) + '    Standard error: ' + str(
        round(stats.sem(np.array(accs)), 3)) )
    plt.savefig(log_path + '/bootstrapping.pdf')


def plot_loss(log_path, train_loss, test_loss, epochs):
    plt.clf()
    plt.figure()
    plt.title('Loss')
    plt.plot(np.arange(epochs), test_loss, label='Test')
    plt.plot(np.arange(epochs), train_loss, label='Train')
    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/train_test_loss.pdf')

def get_other_scores(conf_mat):
                diagonal_sum = conf_mat.trace()
                sum_of_all_elements = conf_mat.sum()
                print('\n')
                print('ACCURACY: ', round(diagonal_sum / float(sum_of_all_elements),2))
                for label in range(3):
                    col = conf_mat[:, label]
                    print('PRECISION CLASS', str(label), '---->', round(conf_mat[label, label] / float(col.sum()),2))
                    row = conf_mat[label, :]
                    print('RECALL CLASS: ', str(label), '---->', round(conf_mat[label, label] / float(row.sum()),2))


def plot_accuracy(log_path, train_acc, test_acc, epochs):
    plt.clf()
    plt.figure()
    plt.title('Accuracy')
    plt.plot(np.arange(epochs), test_acc, label='Test')
    plt.plot(np.arange(epochs), train_acc, label='Train')
    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/train_test_acc.pdf')


def plot_confmat(log_path, confmat):
    plt.clf()
    fig = sns.heatmap(confmat, annot=True, fmt='d')
    conf_mat_cols = 'Predicted Labels'
    conf_mat_rows = 'True labels'
    fig.text(0, 0.96, conf_mat_cols, ha='center', va='center', size=13)
    fig.text(0.06, 0, conf_mat_rows, ha='center', va='center',
             rotation='vertical', size=13)
    plt.savefig(log_path + '/confusion_matrix.pdf')