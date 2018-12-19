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

def plot_loss_multitask(log_path, train_losses, test_losses, epochs):
    struct_losses_test=[]
    struct_losses_train=[]
    solvAcc_losses_test=[]
    solvAcc_losses_train=[]
    flex_losses_test = []
    flex_losses_train = []
    total_losses_test=[]
    total_losses_train=[]

    for l in train_losses:
        struct_losses_train.append(l[0])
        solvAcc_losses_train.append(l[1])
        flex_losses_train.append(l[2])
        total_losses_train.append(l[3])

    for k in test_losses:
        struct_losses_test.append(k[0])
        solvAcc_losses_test.append(k[1])
        flex_losses_test.append(k[2])
        total_losses_test.append(k[3])

    #print(total_losses_test)
    #print(total_losses_train)
    #print(solvAcc_losses_test)
    #print(solvAcc_losses_train)
    #print(struct_losses_test)
    #print(struct_losses_train)

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 4, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Losses over time')

    struct=sns.lineplot(np.arange(epochs), struct_losses_train, label='Train', ax=axes[0])
    struct=sns.lineplot(np.arange(epochs), struct_losses_test, label='Test', ax=axes[0])

    solv=sns.lineplot(np.arange(epochs), solvAcc_losses_train, label='Train', ax=axes[1])
    solv=sns.lineplot(np.arange(epochs), solvAcc_losses_test, label='Test', ax=axes[1])

    flex=sns.lineplot(np.arange(epochs), flex_losses_train, label='Train', ax=axes[2])
    flex=sns.lineplot(np.arange(epochs), flex_losses_test, label='Test', ax=axes[2])

    total = sns.lineplot(np.arange(epochs), total_losses_train, label='Train', ax=axes[3])
    total = sns.lineplot(np.arange(epochs), total_losses_test, label='Test', ax=axes[3])

    struct.set(xlabel=' Structure prediction')
    solv.set(xlabel='Solvent accessibility prediction')
    flex.set(xlabel='Flexibility prediction')
    total.set(xlabel='Multitask prediction')
    struct.set(ylabel='Loss')
    solv.set(ylabel=' ')
    flex.set(ylabel=' ')
    total.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/train_test_loss_multitarget.pdf')


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
    fig, (ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    sns.heatmap(confmat[0], ax=ax3[0], annot=True, fmt='d')
    sns.heatmap(confmat[1], ax=ax3[1], annot=True, fmt='d')
    sns.heatmap(confmat[2], ax=ax3[2], annot=True, fmt='d')

    conf_mat_cols = 'Predicted Labels'
    conf_mat_rows = 'True labels'
    fig.text(0, 0.96, conf_mat_cols, ha='center', va='center', size=13)
    fig.text(0.06, 0, conf_mat_rows, ha='center', va='center',
             rotation='vertical', size=13)
    ax3[0].set_xlabel('Structure')
    ax3[1].set_xlabel('solvAcc')
    ax3[2].set_xlabel('Flex')
    plt.savefig(log_path + '/confusion_matrices.pdf')

def plot_solv_acc_loss(log_path,train_loss_epochs, test_loss_epochs):
    plt.clf()
    plt.figure()
    plt.title('Solvent Accessibility Loss')
    plt.plot(np.arange(len(test_loss_epochs)), test_loss_epochs, label='Test')
    plt.plot(np.arange(len(train_loss_epochs)), train_loss_epochs, label='Train')
    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/solvent_accessibility_loss.pdf')
