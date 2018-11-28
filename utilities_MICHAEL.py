# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:34:25 2018

@author: Michael
"""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    plt.switch_backend('agg')  # GPU is only available via SSH (no display)
import seaborn as sn


def plot_bootstrapping(log_path, matrices):  # matrices=list of confusion matrices
    accs = []
    for i in range(len(matrices)):
        acc = matrices[i].trace() / matrices[i].sum()
        accs.insert(-1, round(acc, 2))

    plt.figure()
    sn.distplot(accs)
    plt.title('Accuracy distribution after last epoch ')
    print('standard deviation: ', np.std(np.array(accs), ddof=1))
    print('standard error: ', stats.sem(np.array(accs)))
    print('mean: ', np.average(np.array(accs)))
    plt.axvline(np.average(np.array(accs)), linestyle='dashed')
    plt.axvline(np.average(np.array(accs)) + np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.axvline(np.average(np.array(accs)) - np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.xlabel('Standard deviation: ' + str(round(np.std(np.array(accs), ddof=1), 3)) + '   Standard error: ' + str(
        round(stats.sem(np.array(accs)), 3)) + ' Mean: ' + str(round(np.average(np.array(accs)), 3)))
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
                print('ACCURACY: ', diagonal_sum / sum_of_all_elements)
                for label in range(3):
                    col = conf_mat[:, label]
                    print('PRECISION CLASS', str(label), '---->', conf_mat[label, label] / col.sum())
                    row = conf_mat[label, :]
                    print('RECALL CLASS: ', str(label), '---->', conf_mat[label, label] / row.sum())


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
    fig = sn.heatmap(confmat, annot=True, fmt='d')
    conf_mat_cols = 'Predicted Labels'
    conf_mat_rows = 'True labels'
    fig.text(0, 0.96, conf_mat_cols, ha='center', va='center', size=13)
    fig.text(0.06, 0, conf_mat_rows, ha='center', va='center',
             rotation='vertical', size=13)
    plt.savefig(log_path + '/confusion_matrix.pdf')


def plot_learning(log_path, eval_summary):
    import matplotlib.pyplot as plt
    if torch.cuda.is_available():
        plt.switch_backend('agg')  # GPU is only available via SSH (no display)
    import seaborn as sn

    plt.clf()  # clear previous figures if already existing

    params = {
        'axes.labelsize': 13,  # increase font size for axis labels
    }
    plt.rc(params)  # apply parameters

    # create figure with subplots
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig4, (ax41) = plt.subplots(2, 1, sharex=True, sharey=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # increase white space between subplots

    for test_set_name, eval_measures in eval_summary.items():  # C1, C2, C3, train
        for measure, data in eval_measures.items():  # loss, accuracy
            epochs = eval_measures['epochs']
            if measure == 'loss':
                ax1.plot(epochs, data, label=test_set_name)
            elif measure == 'accuracy':
                ax2.plot(epochs, data, label=test_set_name)
            elif measure == 'epochs':
                continue  # skip epoch
            elif measure == 'bootstrapped_confmat':
                if (data[-1] != None):
                    print('bootstrapping plotted from:', len(data[-1]))
                    # evaluate bootstrap conf_mats of current epoch
                    accs = []
                    mat = data[-1]
                    # print('MAT', mat)
                    for i in range(len(mat)):
                        acc = mat[i].trace() / mat[i].sum()
                        accs.insert(-1, round(acc, 2))
                    plt.figure()
                    sn.distplot(accs)
                    plt.title('Accuracy distribution after epoch ' + str(epochs[-1]))
                    print('standard deviation: ', np.std(np.array(accs), ddof=1))
                    print('standard error: ', stats.sem(np.array(accs)))
                    print('mean: ', np.average(np.array(accs)))
                    plt.axvline(np.average(np.array(accs)), linestyle='dashed')
                    plt.axvline(np.average(np.array(accs)) + np.std(np.array(accs), ddof=1), linestyle='dashed',
                                color='olive')
                    plt.axvline(np.average(np.array(accs)) - np.std(np.array(accs), ddof=1), linestyle='dashed',
                                color='olive')
                    plt.xlabel('Standard deviation: ' + str(
                        round(np.std(np.array(accs), ddof=1), 3)) + '   Standard error: ' + str(
                        round(stats.sem(np.array(accs)), 3)) + ' Mean: ' + str(round(np.average(np.array(accs)), 3)))
                    plt.savefig(log_path + '/accs.pdf')
            else:  # plot confusion matrix
                data = data[-1]  # always plot only most recent confusion matrix
                # data=data[1:4,1:4]  #ANNAFIX
                if test_set_name == 'Train':
                    sn.heatmap(data, ax=ax41[0], annot=True, fmt='d')
                elif test_set_name == 'Test':
                    sn.heatmap(data, ax=ax41[1], annot=True, fmt='d')

    ax1.legend(loc=2, fontsize='small')  # location code 2: upper left

    ax1.set_ylabel('Loss')  # set axis labels
    ax2.set_ylabel('Accuracy in %')
    # ax2.set_yticks([ax2.get_yticks()*100.0])
    ax2.set_xlabel('Epoch')

    # Axis labels for confusion matrix plot
    conf_mat_cols = 'Predicted Labels after epoch: {}'.format(epochs[-1])
    conf_mat_rows = 'True labels'
    fig4.text(0.5, 0.96, conf_mat_cols, ha='center', va='center', size=13)
    fig4.text(0.06, 0.5, conf_mat_rows, ha='center', va='center',
              rotation='vertical', size=13)

    ax41[0].set_xlabel('Train')  # set axis labels
    ax41[1].set_xlabel('Test')

    # write figures as PDF to disk
    fig1.savefig(log_path + '/loss_acc.pdf', format='pdf')
    fig4.savefig(log_path + '/conf_mat.pdf', format='pdf')

    plt.close(fig1)  # close figure handle
    plt.close(fig4)


def create_logdir(log_path):
    # Try to create a new directory for saving log files
    log_path.mkdir()  # create directory for logging learning progress
    return log_path


def add_progress(eval_summary, test_set_name, loss, epoch,
                 conf_mat=None, y_true=None, y_pred=None, bootstrapped_confmat=None):
    # create dictionary for monitoring loss of various sets while training

    if test_set_name not in eval_summary:  # create non-existing entries
        eval_summary[test_set_name] = {
            'loss': list(),
            'accuracy': list(),
            'epochs': list(),
            'confmat': list(),
            'bootstrapped_confmat': list()
        }

    try:
        acc = accuracy_score(y_true, y_pred)

    except ValueError:  # if (y_true or y_pred) ==None
        acc = 0.

    eval_summary[test_set_name]['epochs'].append(epoch)
    eval_summary[test_set_name]['loss'].append(loss)
    eval_summary[test_set_name]['accuracy'].append(acc)
    eval_summary[test_set_name]['confmat'].append(conf_mat)
    eval_summary[test_set_name]['bootstrapped_confmat'].append(bootstrapped_confmat)


def create_logdir(log_path):
    # Try to create a new directory for saving log files
    try:
        os.mkdir(log_path)  # create directory for logging learning progress
        return log_path
    except FileExistsError:
        print('The directory for saving all log files already exists!' +
              'Choose a different name for your log files.')
        # raise FileExistsError
        pass


def save_performance(log_path, eval_summary, final_flag=False):
    def _get_bootstrap_subsample(conf_mat, subsample_perc=0.1, n_bootstraps=1000):

        n_total = np.sum(conf_mat)
        n_subset = np.round(subsample_perc * n_total).astype(np.int)
        sample_lst = None

        for index in np.ndindex(conf_mat.shape):
            n_reps = conf_mat[index]
            idx_arr = np.asarray(index).reshape(-1, 2)
            tmp_sample_lst = np.repeat(idx_arr, n_reps, axis=0)
            if sample_lst is None:
                sample_lst = tmp_sample_lst
            else:
                sample_lst = np.append(sample_lst, tmp_sample_lst, axis=0)

        prec_table = None
        rec_table = None
        acc_table = None
        for bootstrap_iter in range(n_bootstraps):
            bootstrap_samples = np.random.choice(sample_lst.shape[0], n_subset, replace=False)

            bootstrap_idxs = sample_lst[bootstrap_samples, :]

            bootstrap_confmat = np.ones_like(conf_mat)  # fill with 1s to avoid division by 0 later
            np.add.at(bootstrap_confmat,  # which matrix to add to
                      (bootstrap_idxs[:, 0], bootstrap_idxs[:, 1]),  # indices
                      1  # which value to add to
                      )

            # How many of my predictions (axis=0) were correct? -> Precision in %
            tmp_prec_table = (np.diag(bootstrap_confmat) /
                              np.sum(bootstrap_confmat, axis=0)
                              ).reshape(-1, bootstrap_confmat.shape[0]) * 100
            # How many of my actual samples did I predict correctly? -> Recall in %
            tmp_rec_table = (np.diag(bootstrap_confmat) /
                             np.sum(bootstrap_confmat, axis=1)
                             ).reshape(-1, bootstrap_confmat.shape[0]) * 100
            tmp_acc_table = (np.diag(bootstrap_confmat).sum() /
                             bootstrap_confmat.sum()) * 100
            if prec_table is None:
                prec_table = tmp_prec_table
                rec_table = tmp_rec_table
                acc_table = tmp_acc_table
            else:
                prec_table = np.append(prec_table, tmp_prec_table, axis=0)
                rec_table = np.append(rec_table, tmp_rec_table, axis=0)
                acc_table = np.append(acc_table, tmp_acc_table)

        prec_avg = np.mean(prec_table, axis=0)
        rec_avg = np.mean(rec_table, axis=0)
        acc_avg = np.mean(acc_table).reshape(-1)

        prec_avg_std = np.std(prec_table, axis=0, ddof=1)  # std. dev / std. err
        rec_avg_std = np.std(rec_table, axis=0, ddof=1)  # corrected by population mean
        acc_avg_std = np.std(acc_table, ddof=1).reshape(-1)

        lower_bound = np.ceil(prec_table.shape[0] * 0.025).astype(np.int)
        upper_bound = np.floor(prec_table.shape[0] * 0.975).astype(np.int)
        prec_table.sort(axis=0)
        rec_table.sort(axis=0)
        acc_table.sort()
        prec_confidence = np.vstack((prec_table[lower_bound, None],
                                     prec_table[upper_bound, None])
                                    )
        rec_confidence = np.vstack((rec_table[lower_bound, None],
                                    rec_table[upper_bound, None])
                                   )
        acc_confidence = np.vstack((acc_table[lower_bound],
                                    acc_table[upper_bound])
                                   )

        return {'prec_avg': prec_avg, 'prec_std': prec_avg_std,
                'prec_conf': prec_confidence,
                'rec_avg': rec_avg, 'rec_std': rec_avg_std,
                'rec_conf': rec_confidence,
                'acc_avg': acc_avg, 'acc_std': acc_avg_std,
                'acc_conf': acc_confidence
                }

    '''
        log_path: Directory for saving logged performance
        eval_summary: Dictionary storing performance
            - C1, C2, C3, Train
                - loss, acc, confmat
                    - list holding performance value for each logged epoch
    '''

    txt_f = log_path + '/final_performance.txt'
    np_f = log_path + '/training_process.npz'

    with open(txt_f, 'a+') as f:
        f.write('####################### START #########################\n')
        for test_set_name, eval_measures in eval_summary.items():  # Train & Test
            f.write('\nPerformance for: ' + test_set_name + '\n')
            for measure, data in eval_measures.items():  # loss, acc, confmat
                if measure == 'confmat':
                    f.write(str(data[-1]) + '\n')
                    if final_flag:  # get bootstrap error etc on last epoch

                        performance = _get_bootstrap_subsample(data[-1])
                        for metric in ['prec', 'rec', 'acc']:
                            for class_label in range(data[-1].shape[0]):
                                # plot Qx (accuracy) only once
                                if metric == 'acc' and class_label > 0:
                                    continue

                                f.write('{0} class {1}: {2:.1f} +/- {3:.1f} {4}\n'.format(
                                    metric, class_label,
                                    performance[metric + '_avg'][class_label],
                                    performance[metric + '_std'][class_label],
                                    np.array2string(
                                        performance[metric + '_conf'][:, class_label],
                                        precision=0)
                                )
                                )
                    continue
                elif measure in ['loss', 'acc', 'epoch']:
                    f.write('{0}: {1:.2f}\n'.format(measure, data[-1]))

                if final_flag:  # transform lists to numpy arrays in last epoch
                    eval_summary[test_set_name][measure] = np.asarray(data)
        f.write('####################### END #########################\n')
    if final_flag:
        np.savez(np_f, **eval_summary)

