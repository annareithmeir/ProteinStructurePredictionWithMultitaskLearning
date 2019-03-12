import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

if torch.cuda.is_available():
    plt.switch_backend('agg')
import seaborn as sns

#
# This file handles logging and other smaller tasks
#

def create_logdir(log_path): #Creates the log directory
    splits=str.split(log_path, '/')
    path='/'.join(splits[:-1])
    superpath='/'.join(splits[:2])
    if(not os.path.isdir(superpath)):
        os.mkdir(superpath)
    if (not os.path.isdir(path)):
        os.mkdir(path)
    if( not os.path.isdir(log_path)):
        os.mkdir(log_path)
        return log_path

def writeLogFile(log_path, lr, dssp_mode, model): #Creates a new log file
    f = open(log_path+'/log.txt', 'w')

    f.write(str('LEARNING RATE: '+ str(lr)))
    f.write('\n')
    f.write(str('DSSP MODE :'+ str(dssp_mode)))
    f.write('\n')
    f.write(str('NUMBER OF PARAMETERS IN MODEL :'+ str(count_parameters(model))))
    f.write('\n')
    f.write('Epoch | Train Loss, Test Loss | Train Accuracy, Test Accuracy')
    f.write('\n')

    f.close()

def addProgressToLogFile(log_path, epoch, train_loss, test_loss, train_acc, test_acc): #Logs one epoch in log file
    f=open(log_path+'/log.txt', 'a')
    f.write(str(str(epoch)+' | '+str(train_loss)+', '+str(test_loss)+' | '+str(train_acc)+', '+str(test_acc)))
    f.write('\n')
    f.close()

def writeElapsedTime(log_path, time): #Logs runtime
    f = open(log_path + '/log.txt', 'a')
    f.write(str('TOTAL ELAPSED TIME: '+ str(round(time,3))))
    f.write('\n')
    f.close()

def writeNumberOfParameters(log_path, net, param): #Logs number of learnable parameters of a model
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    f=open(log_path, 'a')
    f.write(str(net+' : '+str(param)))
    f.write('\n')
    f.close()

def write_confmat(log_path, confmat): #Logs the final confusion matrix on testset
    f = open(log_path + '/log.txt', 'a')
    f.write(str('CONFMAT: ' + str(confmat.flatten())))
    f.close()

def write_collected_confmats_8(log_path, collected_confmats): #Logs a list of confusion matrices DSSP8 (For averaging performance per protein later)
    log_path=log_path+'/collected_confmats_log.txt'
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    f=open(log_path, 'w')
    for i in collected_confmats:
        f.write(str(i.flatten()))
        f.write('&')
    f.close()

def write_collected_confmats(log_path, collected_confmats): #Logs a list of confusion matrices DSSP3 (For averaging performance per protein later)
    log_path=log_path+'/collected_confmats_log.txt'
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    f=open(log_path, 'w')
    for i in collected_confmats:
        f.write(str(i.flatten()))
        f.write('\n')
    f.close()

def write_r2(log_path, r2_list): #Logs R2 values for RSA and B-factors
    log_path=log_path+'/R2values.txt'
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    solvAcclist=[]
    flexlist=[]

    for i in r2_list:
        solvAcclist.append(i[0])
        flexlist.append(i[1])

    f=open(log_path, 'w')
    f.write('SOLVACC\n')
    f.write(str('mean:'+str(np.average(np.array(solvAcclist)))+'  std err: '+str(stats.sem(np.array(solvAcclist)))+ '  std dev: '+str(np.std(np.array(solvAcclist), ddof=1))))
    f.write('\nFLEX\n')
    f.write(str('mean:'+str( np.average(np.array(flexlist)))+ '  std err: '+str( stats.sem(np.array(flexlist)))+ '  std dev: '+str(np.std(np.array(flexlist), ddof=1))))
    for i in r2_list:
        f.write(str(i))
        f.write('\n')
    f.close()

def count_parameters(model): #Counts parameters of a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_collected_confmats(log_path): #Plots the accuracy distribution over a list of confusion matrices
    matrices=[]
    with open(log_path + '/collected_confmats_log.txt', 'r') as file:
        data = file.read().split('\n')[:-1]
        for i in data:
            i=i[i.find('[')+1:i.rfind(']')]
            i=i.split()
            i = np.array(list(map(int, i)))
            if (len(i) == 9):
                i=i.reshape(3,3)
            else:
                i=i.reshape(8,8)
            matrices.append(i)

    accs = []
    f_c = []
    f_h = []
    f_e = []
    precs_e=[]
    precs_c=[]
    precs_h=[]
    recs_c=[]
    recs_h=[]
    recs_e=[]

    for i in range(len(matrices)):
        accs.append(matrices[i].trace() / float(matrices[i].sum()))
        prec_c=matrices[i][0,0] / float(matrices[i][:, 0].sum())
        precs_c.append(prec_c)
        rec_c=matrices[i][0,0] / float(matrices[i][0, :].sum())
        recs_c.append(rec_c)
        f_c.append(2*(prec_c*rec_c/float(prec_c+rec_c)))

        prec_h = matrices[i][1, 1] / float(matrices[i][:, 1].sum())
        precs_h.append(prec_h)
        rec_h = matrices[i][1, 1] / float(matrices[i][1, :].sum())
        recs_h.append(rec_h)
        f_h.append(2 * (prec_h * rec_h / float(prec_h + rec_h)))

        prec_e = matrices[i][2, 2] / float(matrices[i][:, 2].sum())
        precs_e.append(prec_e)
        rec_e = matrices[i][2, 2] / float(matrices[i][2, :].sum())
        recs_e.append(rec_e)
        f_e.append(2 * (prec_e * rec_e / float(prec_e + rec_e)))

    precs_c=np.array(precs_c)
    precs_c = precs_c[np.logical_not(np.isnan(precs_c))]
    precs_c=list(precs_c)

    precs_h = np.array(precs_h)
    precs_h = precs_h[np.logical_not(np.isnan(precs_h))]
    precs_h = list(precs_h)

    precs_e = np.array(precs_e)
    precs_e = precs_e[np.logical_not(np.isnan(precs_e))]
    precs_e = list(precs_e)

    recs_c = np.array(recs_c)
    recs_c = recs_c[np.logical_not(np.isnan(recs_c))]
    recs_c = list(recs_c)

    recs_h = np.array(recs_h)
    recs_h = recs_h[np.logical_not(np.isnan(recs_h))]
    recs_h = list(recs_h)

    recs_e = np.array(recs_e)
    recs_e = recs_e[np.logical_not(np.isnan(recs_e))]
    recs_e = list(recs_e)

    f_c = np.array(f_c)
    f_c = f_c[np.logical_not(np.isnan(f_c))]
    f_c = list(f_c)

    f_h = np.array(f_h)
    f_h = f_h[np.logical_not(np.isnan(f_h))]
    f_h = list(f_h)

    f_e = np.array(f_e)
    f_e = f_e[np.logical_not(np.isnan(f_e))]
    f_e = list(f_e)


    plt.figure()
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    sns.distplot(accs)

    with open(log_path + '/collected_confmats_all.txt', 'w') as f:
        f.write('ACCURACY\n')
        f.write(str('std dev: '+ str(np.std(np.array(accs), ddof=1))))
        f.write('\n')
        f.write(str('std error: '+ str(stats.sem(np.array(accs)))))
        f.write('\n')
        f.write(str('mean: '+ str(np.average(np.array(accs)))))
        f.write('\n')

        f.write('PRECISION\n')
        f.write(str('std devs: '+ str(np.std(np.array(precs_c), ddof=1))+','+str(np.std(np.array(precs_h), ddof=1))+','+str(np.std(np.array(precs_e), ddof=1))))
        f.write('\n')
        f.write(str('std errors: '+ str(stats.sem(np.array(precs_c)))+','+str(stats.sem(np.array(precs_h)))+','+str(stats.sem(np.array(precs_e)))))
        f.write('\n')
        f.write(str('means: '+ str(np.average(np.array(precs_c)))+','+str(np.average(np.array(precs_h)))+','+str(np.average(np.array(precs_e)))))
        f.write('\n')

        f.write('RECALL\n')
        f.write(str('std devs: '+ str(np.std(np.array(recs_c), ddof=1))+','+str(np.std(np.array(recs_h), ddof=1))+','+str(np.std(np.array(recs_e), ddof=1))))
        f.write('\n')
        f.write(str('std errors: '+ str(stats.sem(np.array(recs_c)))+','+str(stats.sem(np.array(recs_h)))+','+str(stats.sem(np.array(recs_e)))))
        f.write('\n')
        f.write(str('means: '+ str(np.average(np.array(recs_c)))+','+str(np.average(np.array(recs_h)))+','+str(np.average(np.array(recs_e)))))
        f.write('\n')

        f.write('FMEASURE\n')
        f.write(str('std devs: ' + str(np.std(np.array(f_c), ddof=1)) + ',' + str( np.std(np.array(f_h), ddof=1)) + ',' + str(np.std(np.array(f_e), ddof=1))))
        f.write('\n')
        f.write(str('std errors: ' + str(stats.sem(np.array(f_c))) + ',' + str(stats.sem(np.array(f_h))) + ',' + str(stats.sem(np.array(f_e)))))
        f.write('\n')
        f.write(str('means: ' + str(np.average(np.array(f_c))) + ',' + str(np.average(np.array(f_h))) + ',' + str(np.average(np.array(f_e)))))
        f.write('\n')


    plt.axvline(np.average(np.array(accs)), linestyle='dashed')
    plt.axvline(np.average(np.array(accs)) + np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.axvline(np.average(np.array(accs)) - np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.savefig(log_path + '/collected_confmats.pdf')

def plot_collected_confmats8(log_path): #Same as above but for DSSP8
    matrices=[]
    with open(log_path + '/collected_confmats_log.txt', 'r') as file:
        data = file.read().split('&')
        data=data[:-1]
        for l in data:
            i=l[1:-1]
            i=i.split()
            i = np.array(list(map(int, i)))
            i=i.reshape(8,8)
            matrices.append(i)

    accs = []
    f_h = []
    f_e = []
    f_i = []
    f_s = []
    f_t = []
    f_g = []
    f_b = []
    f_none = []
    precs_e=[]
    precs_i=[]
    precs_h=[]
    precs_s = []
    precs_t = []
    precs_g = []
    precs_b = []
    precs_none = []
    recs_i=[]
    recs_h=[]
    recs_e=[]
    recs_i = []
    recs_s = []
    recs_t = []
    recs_g = []
    recs_b = []
    recs_none = []

    for i in range(len(matrices)):
        accs.append(matrices[i].trace() / float(matrices[i].sum()))

        prec_h=matrices[i][0,0] / float(matrices[i][:, 0].sum())
        precs_h.append(prec_h)
        rec_h=matrices[i][0,0] / float(matrices[i][0, :].sum())
        recs_h.append(rec_h)
        f_h.append(2*(prec_h*rec_h/float(prec_h+rec_h)))

        prec_i = matrices[i][2, 2] / float(matrices[i][:, 2].sum())
        precs_i.append(prec_i)
        rec_i = matrices[i][2, 2] / float(matrices[i][2, :].sum())
        recs_i.append(rec_i)
        f_i.append(2 * (prec_i * rec_i / float(prec_i + rec_i)))

        prec_e = matrices[i][1,1] / float(matrices[i][:, 1].sum())
        precs_e.append(prec_e)
        rec_e = matrices[i][1, 1] / float(matrices[i][1, :].sum())
        recs_e.append(rec_e)
        f_e.append(2 * (prec_e * rec_e / float(prec_e + rec_e)))

        prec_s = matrices[i][3, 3] / float(matrices[i][:, 3].sum())
        precs_s.append(prec_s)
        rec_s = matrices[i][3, 3] / float(matrices[i][3, :].sum())
        recs_s.append(rec_s)
        f_s.append(2 * (prec_s * rec_s / float(prec_s + rec_s)))

        prec_t = matrices[i][4, 4] / float(matrices[i][:, 4].sum())
        precs_t.append(prec_t)
        rec_t = matrices[i][4, 4] / float(matrices[i][4, :].sum())
        recs_t.append(rec_t)
        f_t.append(2 * (prec_t * rec_t / float(prec_t + rec_t)))

        prec_g = matrices[i][5, 5] / float(matrices[i][:, 5].sum())
        precs_g.append(prec_g)
        rec_g = matrices[i][5, 5] / float(matrices[i][5, :].sum())
        recs_g.append(rec_g)
        f_g.append(2 * (prec_g * rec_g / float(prec_g + rec_g)))

        prec_b = matrices[i][6, 6] / float(matrices[i][:, 6].sum())
        precs_b.append(prec_b)
        rec_b = matrices[i][6, 6] / float(matrices[i][6, :].sum())
        recs_b.append(rec_b)
        f_b.append(2 * (prec_b * rec_b / float(prec_b + rec_b)))

        prec_none = matrices[i][7, 7] / float(matrices[i][:, 7].sum())
        precs_none.append(prec_none)
        rec_none = matrices[i][7, 7] / float(matrices[i][7, :].sum())
        recs_none.append(rec_none)
        f_none.append(2 * (prec_none * rec_none / float(prec_none + rec_none)))


    precs_i=np.array(precs_i)
    precs_i = precs_i[np.logical_not(np.isnan(precs_i))]
    precs_i=list(precs_i)

    precs_h = np.array(precs_h)
    precs_h = precs_h[np.logical_not(np.isnan(precs_h))]
    precs_h = list(precs_h)

    precs_e = np.array(precs_e)
    precs_e = precs_e[np.logical_not(np.isnan(precs_e))]
    precs_e = list(precs_e)

    precs_s = np.array(precs_s)
    precs_s = precs_s[np.logical_not(np.isnan(precs_s))]
    precs_s = list(precs_s)

    precs_t = np.array(precs_t)
    precs_t = precs_t[np.logical_not(np.isnan(precs_t))]
    precs_t = list(precs_t)

    precs_g = np.array(precs_g)
    precs_g = precs_g[np.logical_not(np.isnan(precs_g))]
    precs_g = list(precs_g)

    precs_b = np.array(precs_b)
    precs_b = precs_b[np.logical_not(np.isnan(precs_b))]
    precs_b = list(precs_b)

    precs_none = np.array(precs_none)
    precs_none = precs_none[np.logical_not(np.isnan(precs_none))]
    precs_none = list(precs_none)

    recs_i = np.array(recs_i)
    recs_i = recs_i[np.logical_not(np.isnan(recs_i))]
    recs_i = list(recs_i)

    recs_h = np.array(recs_h)
    recs_h = recs_h[np.logical_not(np.isnan(recs_h))]
    recs_h = list(recs_h)

    recs_e = np.array(recs_e)
    recs_e = recs_e[np.logical_not(np.isnan(recs_e))]
    recs_e = list(recs_e)

    recs_s = np.array(recs_s)
    recs_s = recs_s[np.logical_not(np.isnan(recs_s))]
    recs_s = list(recs_s)

    recs_t = np.array(recs_t)
    recs_t = recs_t[np.logical_not(np.isnan(recs_t))]
    recs_t = list(recs_t)

    recs_g = np.array(recs_g)
    recs_g = recs_g[np.logical_not(np.isnan(recs_g))]
    recs_g = list(recs_g)

    recs_b = np.array(recs_b)
    recs_b = recs_b[np.logical_not(np.isnan(recs_b))]
    recs_b = list(recs_b)

    recs_none = np.array(recs_none)
    recs_none = recs_none[np.logical_not(np.isnan(recs_none))]
    recs_none = list(recs_none)

    f_i = np.array(f_i)
    f_i = f_i[np.logical_not(np.isnan(f_i))]
    f_i = list(f_i)

    f_h = np.array(f_h)
    f_h = f_h[np.logical_not(np.isnan(f_h))]
    f_h = list(f_h)

    f_e = np.array(f_e)
    f_e = f_e[np.logical_not(np.isnan(f_e))]
    f_e = list(f_e)

    f_s = np.array(f_s)
    f_s = f_s[np.logical_not(np.isnan(f_s))]
    f_s = list(f_s)

    f_t = np.array(f_t)
    f_t = f_t[np.logical_not(np.isnan(f_t))]
    f_t = list(f_t)

    f_g = np.array(f_g)
    f_g = f_g[np.logical_not(np.isnan(f_g))]
    f_g = list(f_g)

    f_b = np.array(f_b)
    f_b = f_b[np.logical_not(np.isnan(f_b))]
    f_b = list(f_b)

    f_none = np.array(f_none)
    f_none = f_none[np.logical_not(np.isnan(f_none))]
    f_none = list(f_none)


    plt.figure()
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    sns.distplot(accs)

    with open(log_path + '/collected_confmats_all.txt', 'w') as f:
        f.write('ACCURACY\n')
        f.write(str('std dev: '+ str(np.std(np.array(accs), ddof=1))))
        f.write('\n')
        f.write(str('std error: '+ str(stats.sem(np.array(accs)))))
        f.write('\n')
        f.write(str('mean: '+ str(np.average(np.array(accs)))))
        f.write('\n')

        f.write('PRECISION\n')
        f.write(str('std devs: '+ str(np.std(np.array(precs_h), ddof=1))+','
                    +str(np.std(np.array(precs_e), ddof=1))+','
                    + str(np.std(np.array(precs_i), ddof=1)) + ','
                    + str(np.std(np.array(precs_s), ddof=1)) + ','
                    + str(np.std(np.array(precs_t), ddof=1)) + ','
                    + str(np.std(np.array(precs_g), ddof=1)) + ','
                    + str(np.std(np.array(precs_b), ddof=1)) + ','
                    + str(np.std(np.array(precs_none), ddof=1))))
        f.write('\n')
        f.write(str('std errors: '+ str(stats.sem(np.array(precs_h)))+','
                    +str(stats.sem(np.array(precs_e)))+','
                    +str(stats.sem(np.array(precs_i)))+','
                    + str(stats.sem(np.array(precs_s))) + ','
                    + str(stats.sem(np.array(precs_t))) + ','
                    + str(stats.sem(np.array(precs_g))) + ','
                    + str(stats.sem(np.array(precs_b))) + ','
                    + str(stats.sem(np.array(precs_none)))))
        f.write('\n')
        f.write(str('means: '+ str(np.average(np.array(precs_h)))+','
                    +str(np.average(np.array(precs_e)))+','
                    + str(np.average(np.array(precs_i))) + ','
                    + str(np.average(np.array(precs_s))) + ','
                    + str(np.average(np.array(precs_t))) + ','
                    + str(np.average(np.array(precs_g))) + ','
                    + str(np.average(np.array(precs_b))) + ','
                    +str(np.average(np.array(precs_none)))))
        f.write('\n')

        f.write('RECALL\n')
        f.write(str('std devs: '+ str(np.std(np.array(recs_h), ddof=1))+','
                    +str(np.std(np.array(recs_e), ddof=1))+','
                    + str(np.std(np.array(recs_i), ddof=1)) + ','
                    + str(np.std(np.array(recs_s), ddof=1)) + ','
                    + str(np.std(np.array(recs_t), ddof=1)) + ','
                    + str(np.std(np.array(recs_g), ddof=1)) + ','
                    + str(np.std(np.array(recs_b), ddof=1)) + ','
                    +str(np.std(np.array(recs_none), ddof=1))))
        f.write('\n')
        f.write(str('std errors: '+ str(stats.sem(np.array(recs_h)))+','
                    +str(stats.sem(np.array(recs_e)))+','
                    + str(stats.sem(np.array(recs_i))) + ','
                    + str(stats.sem(np.array(recs_s))) + ','
                    + str(stats.sem(np.array(recs_t))) + ','
                    + str(stats.sem(np.array(recs_g))) + ','
                    + str(stats.sem(np.array(recs_b))) + ','
                    +str(stats.sem(np.array(recs_none)))))
        f.write('\n')
        f.write(str('means: '+ str(np.average(np.array(recs_h)))+','
                    +str(np.average(np.array(recs_e)))+','
                    + str(np.average(np.array(recs_i))) + ','
                    + str(np.average(np.array(recs_s))) + ','
                    + str(np.average(np.array(recs_t))) + ','
                    + str(np.average(np.array(recs_g))) + ','
                    + str(np.average(np.array(recs_b))) + ','
                    +str(np.average(np.array(recs_none)))))
        f.write('\n')

        f.write('FMEASURE\n')
        f.write(str('std devs: ' + str(np.std(np.array(f_h), ddof=1)) + ','
                    + str( np.std(np.array(f_e), ddof=1)) + ','
                    + str(np.std(np.array(f_i), ddof=1)) + ','
                    + str(np.std(np.array(f_s), ddof=1)) + ','
                    + str(np.std(np.array(f_t), ddof=1)) + ','
                    + str(np.std(np.array(f_g), ddof=1)) + ','
                    + str(np.std(np.array(f_b), ddof=1)) + ','
                    + str(np.std(np.array(f_none), ddof=1))))
        f.write('\n')
        f.write(str('std errors: ' + str(stats.sem(np.array(f_h))) + ','
                    + str(stats.sem(np.array(f_e))) + ','
                    + str(stats.sem(np.array(f_i))) + ','
                    + str(stats.sem(np.array(f_s))) + ','
                    + str(stats.sem(np.array(f_t))) + ','
                    + str(stats.sem(np.array(f_g))) + ','
                    + str(stats.sem(np.array(f_b))) + ','
                    + str(stats.sem(np.array(f_none)))))
        f.write('\n')
        f.write(str('means: ' + str(np.average(np.array(f_h))) + ','
                    + str(np.average(np.array(f_e))) + ','
                    + str(np.average(np.array(f_i))) + ','
                    + str(np.average(np.array(f_s))) + ','
                    + str(np.average(np.array(f_t))) + ','
                    + str(np.average(np.array(f_g))) + ','
                    + str(np.average(np.array(f_b))) + ','
                    + str(np.average(np.array(f_none)))))
        f.write('\n')

    plt.axvline(np.average(np.array(accs)), linestyle='dashed')
    plt.axvline(np.average(np.array(accs)) + np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.axvline(np.average(np.array(accs)) - np.std(np.array(accs), ddof=1), linestyle='dashed', color='olive')
    plt.savefig(log_path + '/collected_confmats.pdf')

def plot_loss(log_path, train_loss, test_loss, epochs): #Plots loss over epochs
    plt.clf()
    plt.figure()
    plt.title('Loss')
    plt.plot(np.arange(epochs), test_loss, label='Test')
    plt.plot(np.arange(epochs), train_loss, label='Train')
    plt.legend(loc=2, fontsize='large<')
    plt.savefig(log_path + '/train_test_loss.pdf')

def plot_loss_multitask(log_path, train_losses, test_losses, epochs): #Plots losses for single tasks in multi-task learning
    struct3_losses_test=[]
    struct3_losses_train=[]
    solvAcc_losses_test=[]
    solvAcc_losses_train=[]
    flex_losses_test = []
    flex_losses_train = []
    total_losses_test=[]
    total_losses_train=[]

    for l in train_losses:
        struct3_losses_train.append(l[0])
        solvAcc_losses_train.append(l[1])
        flex_losses_train.append(l[2])
        total_losses_train.append(l[3])

    for k in test_losses:
        struct3_losses_test.append(k[0])
        solvAcc_losses_test.append(k[1])
        flex_losses_test.append(k[2])
        total_losses_test.append(k[3])

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 4, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Losses over time')

    struct3_train=sns.lineplot(np.arange(epochs), struct3_losses_train, label='Train', ax=axes[0])
    struct3_test=sns.lineplot(np.arange(epochs), struct3_losses_test, label='Test', ax=axes[0])

    solv_train=sns.lineplot(np.arange(epochs), solvAcc_losses_train, label='Train', ax=axes[1])
    solv_test=sns.lineplot(np.arange(epochs), solvAcc_losses_test, label='Test', ax=axes[1])

    flex_train=sns.lineplot(np.arange(epochs), flex_losses_train, label='Train', ax=axes[2])
    flex_test=sns.lineplot(np.arange(epochs), flex_losses_test, label='Test', ax=axes[2])

    total_train = sns.lineplot(np.arange(epochs), total_losses_train, label='Train', ax=axes[3])
    total_test = sns.lineplot(np.arange(epochs), total_losses_test, label='Test', ax=axes[3])

    struct3_train.set(xlabel=' Structure 3 prediction')
    solv_train.set(xlabel='Solvent accessibility prediction')
    flex_train.set(xlabel='Flexibility prediction')
    total_train.set(xlabel='Multitask prediction')
    struct3_train.set(ylabel='Loss')
    solv_train.set(ylabel=' ')
    flex_train.set(ylabel=' ')
    total_train.set(ylabel=' ')

    plt.legend(loc=2, fontsize='large')
    plt.savefig(log_path + '/train_test_loss_multitarget.pdf')

def get_other_scores_fromfile(log_path): #Computes different performance scores
    with open(log_path + '/log.txt', 'r') as file:
        data = file.read().split('\n')
        if(data[-1][:7]=='CONFMAT'):
            data = data[-1]
        else:
            data=data[-6:]
            acc=''
            for l in data:
                acc+=l
            data=acc

        data = data[data.find('[') + 1:data.rfind(']')]
        data = data.split()
        data = np.array(list(map(int, data)))

    if (len(data) == 9):
        conf_mat = data.reshape(3, 3)
    else:
        print(data)
        conf_mat = data.reshape(8, 8)

    print(conf_mat)

    diagonal_sum = conf_mat.trace()
    sum_of_all_elements = conf_mat.sum()
    print('\n')
    print('ACCURACY: ', round(diagonal_sum / float(sum_of_all_elements), 2))
    prec_macroavg=0
    rec_macroavg=0
    for label in range(conf_mat.shape[0]):
        col = conf_mat[:, label]
        row = conf_mat[label, :]
        precision = conf_mat[label, label] / float(col.sum())
        prec_macroavg+=precision
        recall = conf_mat[label, label] / float(row.sum())
        rec_macroavg+=recall
        f1 = 2 * (recall * precision) / (recall + precision)
        print('PRECISION CLASS', str(label), '---->', round(precision, 2))
        print('RECALL CLASS: ', str(label), '---->', round(recall, 2))
        print('f1 SCORE: ', str(label), '---->', round(f1, 2))
    prec_macroavg=prec_macroavg/float(conf_mat.shape[0])
    rec_macroavg = rec_macroavg / float(conf_mat.shape[0])
    print('Macro avg precision:',round(prec_macroavg,2),'  macro avg recall:',round(rec_macroavg,2))


    pooled=[]
    for label in range(conf_mat.shape[0]):
        pooled_matrix=np.zeros((2,2))
        col = conf_mat[:, label]
        row = conf_mat[label, :]
        pooled_matrix[0, 0]=col[label] #TP
        pooled_matrix[0, 1] = row.sum() - row[label] #FP
        pooled_matrix[1, 0] = col.sum() - col[label]  # FN
        pooled_matrix[1, 1] = conf_mat.sum()-row.sum() - col.sum() + row[label]  # FP

        pooled.append(pooled_matrix)

        print('####')
        print(label)
        print(pooled_matrix)

    pooled_sum=np.zeros((2,2))
    for m in pooled:
        pooled_sum += m
    print(pooled_sum)

    precpooled = pooled_sum[0,0] / float(pooled_sum[0,0]+pooled_sum[1,0])
    recpooled = pooled_sum[0,0]  / float(pooled_sum[0,0]+pooled_sum[0,1])
    print('micro avg precision:', round(precpooled,2), 'micro avg recall:', round(recpooled,2))
    print('---')
    print('F1 macro: ',2 * (rec_macroavg * prec_macroavg) / (rec_macroavg + prec_macroavg))
    print('F1 micro: ', 2 * (recpooled * precpooled) / (recpooled + precpooled))
    print('---')

def plot_accuracy(log_path, train_acc, test_acc, epochs): #Plots accuracy over epochs
    sns.set()
    sns.set_style('white')
    plt.clf()
    plt.figure()
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    l=str.split(str.split(log_path, '/')[-1],'_')
    plt.suptitle('Q'+l[2]+' Accuracy of '+l[0]+' with '+l[1]+ ' as input')
    plt.plot(np.arange(epochs), test_acc, label='Test')
    plt.plot(np.arange(epochs), train_acc, label='Train')
    plt.legend(loc='lower right', fontsize='large')
    plt.savefig(log_path + '/train_test_structure_acc.pdf')

def plot_confmat(log_path): #Plots a confusion matrix
    with open(log_path + '/log.txt', 'r') as file:
        data = file.read().split('\n')
        if (data[-1][:7] == 'CONFMAT'):
            data = data[-1]
        else:
            data = data[-6:]
            acc = ''
            for l in data:
                acc += l
            data = acc
        data=data[data.find('[')+1:data.rfind(']')]
        data=data.split()
        data = np.array(list(map(int, data)))

    if(len(data)==9):
        confmat=data.reshape(3,3)

        d = pd.DataFrame(confmat, index=['C', 'H', 'E'], columns=['C', 'H', 'E'])
    else:
        confmat = data.reshape(8, 8)
        d = pd.DataFrame(confmat, index=['H', 'E', 'I','S','T','G','B','-'], columns=['H', 'E', 'I','S','T','G','B','-'])

    plt.clf()
    sns.set_context('poster')

    if(confmat.shape[0]==8):
        plt.figure(figsize=(18,14))
    else:
        plt.figure()
    sns.set_context('poster')
    plt.tick_params(axis='x', top=False, bottom=False)
    plt.tick_params(axis='y', right=False, left=False)

    conf_mat_cols = 'Predicted Labels'
    conf_mat_rows = 'True Labels'

    if (confmat.shape[0] == 3):
        fig = sns.heatmap(d, annot=confmat, fmt='.0f', cmap="YlGnBu_r", vmin=0, vmax=41993, cbar=True)
        fig.text(1.5, -0.2, conf_mat_cols, ha='center', va='center', size=25)
        fig.text(-0.4, 1.5, conf_mat_rows, ha='center', va='center',rotation='vertical', size=25)
    else:
        fig = sns.heatmap(d, annot=confmat, fmt='.0f', cmap="YlGnBu_r", cbar=True, annot_kws={'fontsize':35})
        fig.text(4.0, -0.25, conf_mat_cols, ha='center', va='center', size=25)
        fig.text(-0.6, 4.0, conf_mat_rows, ha='center', va='center',rotation='vertical', size=25)
    plt.savefig(log_path + '/structure_confusion_matrix.pdf')

#Given a input representation, plots a comparison of different networks
def plotComparisonForInputType(log_path, input_type, multi_mode):
    sns.set()
    sns.set_style('white')
    plt.clf()
    plt.figure()
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    dirs=[]
    dictionary=dict()

    for (dirpath, dirnames, filenames) in os.walk(log_path):
        for name in dirnames:
            splits = name.split('_')
            if (splits[1] == input_type):
                dirs.append(str(log_path + name))
        break

    for dir in dirs:
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-2]
            for epoch in data:
                epochacc=epoch.split(' | ')[-1] #train acc, test acc
                epochacc=epochacc.split(', ')
                train_acc.append(str(epochacc[0]))
                test_acc.append(str(epochacc[1]))

                epochloss = epoch.split(' | ')[-2]  # train acc, test acc
                epochloss = epochloss.split(', ')
                train_loss.append(str(epochloss[3][:-1]))
                test_loss.append(str(epochloss[-1][:-1]))

            train_acc = list(map(float, train_acc))
            test_acc = list(map(float, test_acc))
            train_loss = list(map(float, train_loss))
            test_loss = list(map(float, test_loss))
            assert ( len(train_acc)== len(test_acc)==len(train_loss)== len(test_loss))
            tmp={dir : [train_loss, test_loss, train_acc, test_acc]}
            dictionary.update(tmp)

    plt.clf()
    sns.set_context('poster')
    plt.figure()
    sns.set_context('poster')
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        train_acc = dictionary[dir][2]
        test_acc = dictionary[dir][3]

        label = dir.split('/')[-1].split('_')[0]
        if (label == 'biLSTM'):
            label = 'BiLSTM'

        train_acc = [i * 100 for i in train_acc]
        test_acc = [i * 100 for i in test_acc]
        df_train[label] = train_acc
        df_test[label] = test_acc

    df_train['x'] = range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'CNN', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'DenseCNN', data=df_train, linewidth=4)
    axes[0].plot('x', 'LSTM', data=df_train, linewidth=4)
    axes[0].plot('x', 'BiLSTM', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'Hybrid', data=df_train, linewidth=4)
    axes[0].plot('x', 'DenseHybrid', data=df_train, linewidth=4)

    axes[1].plot('x', 'CNN', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'DenseCNN', data=df_test, linewidth=4)
    axes[1].plot('x', 'LSTM', data=df_test, linewidth=4)
    axes[1].plot('x', 'BiLSTM', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'Hybrid', data=df_test, linewidth=4)
    axes[1].plot('x', 'DenseHybrid', data=df_test, linewidth=4)

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Accuracy in %', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='lower right', fontsize='large')
    axes[1].legend(loc='lower right', fontsize='large')
    plt.savefig(log_path + 'acc_comparison_'+input_type+'_'+multi_mode+'.pdf')
    plt.close()
    plt.clf()
    sns.set_context('poster')
    plt.figure()
    sns.set_context('poster')
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        label = dir.split('/')[-1].split('_')[0]
        if (label == 'biLSTM'):
            label = 'BiLSTM'
        train_loss = dictionary[dir][0]
        test_loss = dictionary[dir][1]

        df_train[label] = train_loss
        df_test[label] = test_loss

    df_train['x'] = range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'CNN', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'DenseCNN', data=df_train, linewidth=4)
    axes[0].plot('x', 'LSTM', data=df_train, linewidth=4)
    axes[0].plot('x', 'BiLSTM', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'Hybrid', data=df_train, linewidth=4)
    axes[0].plot('x', 'DenseHybrid', data=df_train, linewidth=4)

    axes[1].plot('x', 'CNN', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'DenseCNN', data=df_test, linewidth=4)
    axes[1].plot('x', 'LSTM', data=df_test, linewidth=4)
    axes[1].plot('x', 'BiLSTM', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'Hybrid', data=df_test, linewidth=4)
    axes[1].plot('x', 'DenseHybrid', data=df_test, linewidth=4)

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Loss', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='upper right', fontsize='large')
    axes[1].legend(loc='upper right', fontsize='large')
    plt.savefig(log_path + 'loss_comparison_' + input_type + '_' + multi_mode + '.pdf')
    plt.close()


#Given a network, plots a comparison of different input types
def plotComparisonForNetworkType(log_path, network_type):
    sns.set()
    sns.set_style('white')

    plt.clf()
    plt.figure()

    dirs=[]
    dictionary=dict()

    for (dirpath, dirnames, filenames) in os.walk(log_path):
        for name in dirnames:
            splits = name.split('_')
            if (splits[0] == network_type):
                dirs.append(str(log_path + name))
        break

    for dir in dirs:
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-2]
            for epoch in data:
                epochacc=epoch.split(' | ')[-1] #train acc, test acc
                epochacc=epochacc.split(', ')
                train_acc.append(str(epochacc[0]))
                test_acc.append(str(epochacc[1]))

                epochloss = epoch.split(' | ')[-2]  # train acc, test acc
                epochloss = epochloss.split(', ')
                train_loss.append(str(epochloss[3][:-1]))
                test_loss.append(str(epochloss[-1][:-1]))

            train_acc = list(map(float, train_acc))
            test_acc = list(map(float, test_acc))
            train_loss = list(map(float, train_loss))
            test_loss = list(map(float, test_loss))
            assert ( len(train_acc)== len(test_acc)==len(train_loss)== len(test_loss))
            tmp={dir : [train_loss, test_loss, train_acc, test_acc]}
            dictionary.update(tmp)

    plt.clf()
    sns.set_context('poster')
    plt.figure()
    sns.set_context('poster')
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex='all', sharey='all')
    sns.set_context('poster')

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        train_acc = dictionary[dir][2]
        test_acc = dictionary[dir][3]

        label = dir.split('/')[-1].split('_')[1]
        if (label == 'protvec+scoringmatrix'):
            label = 'PSSM+ProtVec'
        if (label == 'pssm'):
            label = 'PSSM'
        if (label == 'protvecevolutionary'):
            label = 'ProtVec (MSA)'
        if (label == 'protvec'):
            label = 'ProtVec'
        if (label == '1hot'):
            label = 'One-hot'

        train_acc = [i * 100 for i in train_acc]
        test_acc = [i * 100 for i in test_acc]
        df_train[label]=train_acc
        df_test[label]=test_acc

    df_train['x']=range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'One-hot', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'ProtVec', data=df_train,linewidth=4)
    axes[0].plot('x', 'ProtVec (MSA)', data=df_train, linewidth=4)
    axes[0].plot('x', 'PSSM', data=df_train,linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'PSSM+ProtVec', data=df_train,linewidth=4)

    axes[1].plot('x', 'One-hot', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'ProtVec', data=df_test, linewidth=4)
    axes[1].plot('x', 'ProtVec (MSA)', data=df_test, linewidth=4)
    axes[1].plot('x', 'PSSM', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'PSSM+ProtVec', data=df_test, linewidth=4)

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Accuracy in %', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='lower right', fontsize='large')
    axes[1].legend(loc='lower right', fontsize='large')
    plt.savefig(log_path + 'acc_comparison_'+network_type+'.pdf')
    plt.close()

    plt.clf()
    plt.figure()


    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex='all', sharey='all')
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        train_acc = dictionary[dir][0]
        test_acc = dictionary[dir][1]

        label = dir.split('/')[-1].split('_')[1]
        if (label == 'protvec+scoringmatrix'):
            label = 'PSSM+ProtVec'
        if (label == 'pssm'):
            label = 'PSSM'
        if (label == 'protvecevolutionary'):
            label = 'ProtVec (MSA)'
        if (label == 'protvec'):
            label = 'ProtVec'
        if (label == '1hot'):
            label = 'One-hot'

        df_train[label]=train_acc
        df_test[label]=test_acc

    df_train['x']=range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'One-hot', data=df_train, linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'ProtVec', data=df_train,linewidth=4)
    axes[0].plot('x', 'ProtVec (MSA)', data=df_train, linewidth=4)
    axes[0].plot('x', 'PSSM', data=df_train,linewidth=4, linestyle='dashed')
    axes[0].plot('x', 'PSSM+ProtVec', data=df_train,linewidth=4)

    axes[1].plot('x', 'One-hot', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'ProtVec', data=df_test, linewidth=4)
    axes[1].plot('x', 'ProtVec (MSA)', data=df_test, linewidth=4)
    axes[1].plot('x', 'PSSM', data=df_test, linewidth=4, linestyle='dashed')
    axes[1].plot('x', 'PSSM+ProtVec', data=df_test, linewidth=4)

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Loss', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='upper right', fontsize='large')
    axes[1].legend(loc='upper right', fontsize='large')
    plt.savefig(log_path + 'loss_comparison_'+network_type+'.pdf')
    plt.close()

#Given a input representation and network, plots a comparison of different multi-task types
def plotComparisonForMultitargetType(log_path, network_type, input_type, target_types):
    sns.set()
    sns.set_style('white')
    plt.clf()
    plt.figure()
    dirs=[]
    multimodes=['structure', 'multi2', 'multi3','multi4']
    dictionary=dict()

    for tt in target_types:
        for mt in multimodes:
            if(os.path.exists(str(log_path+mt+'/'+str(tt)+'/'))):
                for (dirpath, dirnames, filenames) in os.walk(str(log_path+mt+'/'+str(tt)+'/')):
                    for name in dirnames:
                        splits=name.split('_')

                        if(splits[0]==network_type and splits[1] in input_type):
                            dirs.append(str(log_path+mt+'/'+str(tt)+'/'+name))


    for dir in dirs:
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-2]
            for epoch in data:
                epochacc=epoch.split(' | ')[-1] #train acc, test acc
                epochacc=epochacc.split(', ')
                train_acc.append(str(epochacc[0]))
                test_acc.append(str(epochacc[1]))
                epochloss = epoch.split(' | ')[-2]  # train acc, test acc
                epochloss = epochloss.split(', ')
                train_loss.append(str(epochloss[3][:-1]))
                test_loss.append(str(epochloss[-1][:-1]))

            train_acc = list(map(float, train_acc))
            test_acc = list(map(float, test_acc))
            train_loss = list(map(float, train_loss))
            test_loss = list(map(float, test_loss))
            assert ( len(train_acc)== len(test_acc)==len(train_loss)== len(test_loss))
            tmp={dir : [train_loss, test_loss, train_acc, test_acc]}
            dictionary.update(tmp)

    plt.clf()
    sns.set_context('poster')
    plt.figure()
    sns.set_context('poster')
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    sns.set_context('poster')

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        train_acc = dictionary[dir][2]
        test_acc = dictionary[dir][3]

        label = dir.split('/')[-3]
        if (label == 'multi2'):
            label = 'DSSP3 & RSA'
        elif (label == 'multi3'):
            label = 'DSSP3 & RSA & B-Factors'
        elif (label == 'multi4'):
            label = 'DSSP3 & DSSP8 & RSA & B-Factors'

        train_acc = [i * 100 for i in train_acc]
        test_acc = [i * 100 for i in test_acc]
        df_train[label] = train_acc
        df_test[label] = test_acc

    df_train['x'] = range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'DSSP3 & RSA', data=df_train, linewidth=4)
    axes[0].plot('x', 'DSSP3 & RSA & B-Factors', data=df_train, linewidth=4)
    axes[0].plot('x', 'DSSP3 & DSSP8 & RSA & B-Factors', data=df_train, linewidth=4, linestyle='dashed')

    axes[1].plot('x', 'DSSP3 & RSA', data=df_test, linewidth=4)
    axes[1].plot('x', 'DSSP3 & RSA & B-Factors', data=df_test, linewidth=4)
    axes[1].plot('x', 'DSSP3 & DSSP8 & RSA & B-Factors', data=df_test, linewidth=4, linestyle='dashed')

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Accuracy in %', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='lower right', fontsize='large')
    axes[1].legend(loc='lower right', fontsize='large')
    plt.savefig(log_path + 'acc_comparison_'+network_type+'_multimodes.pdf')
    plt.close()
    plt.clf()
    sns.set_context('poster')
    plt.figure()
    sns.set_context('poster')
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for dir in dictionary:
        train_acc = dictionary[dir][0]
        test_acc = dictionary[dir][1]

        label = dir.split('/')[-3]
        if (label == 'multi2'):
            label = 'DSSP3 & RSA'
        elif (label == 'multi3'):
            label = 'DSSP3 & RSA & B-Factors'
        elif (label == 'multi4'):
            label = 'DSSP3 & DSSP8 & RSA & B-Factors'

        df_train[label] = train_acc
        df_test[label] = test_acc

    df_train['x'] = range(100)
    df_test['x'] = range(100)

    axes[0].plot('x', 'DSSP3 & RSA', data=df_train, linewidth=4)
    axes[0].plot('x', 'DSSP3 & RSA & B-Factors', data=df_train, linewidth=4)
    axes[0].plot('x', 'DSSP3 & DSSP8 & RSA & B-Factors', data=df_train, linewidth=4, linestyle='dashed')

    axes[1].plot('x', 'DSSP3 & RSA', data=df_test, linewidth=4)
    axes[1].plot('x', 'DSSP3 & RSA & B-Factors', data=df_test, linewidth=4)
    axes[1].plot('x', 'DSSP3 & DSSP8 & RSA & B-Factors', data=df_test, linewidth=4, linestyle='dashed')

    axes[0].tick_params(axis='x', top=False)
    axes[0].tick_params(axis='y', right=False)

    axes[1].tick_params(axis='x', top=False)
    axes[1].tick_params(axis='y', right=False)

    axes[0].set_xlabel(' Training Epochs', fontsize=30)
    axes[1].set_xlabel('Testing Epochs', fontsize=30)
    axes[0].set_ylabel('Accuracy in %', fontsize=30)
    axes[1].set(ylabel=' ')

    axes[0].legend(loc='upper right', fontsize='large')
    axes[1].legend(loc='upper right', fontsize='large')
    plt.savefig(log_path + '/loss_comparison_'+network_type+'_multimodes.pdf')
    plt.close()

def writeSolvAccFlex(log_path,predarray_sa, truearray_sa, predarray_fl, truearray_fl): #Logs RSA and B-factor predictions
    log_path = log_path + '/multitask_log.txt'
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    f = open(log_path, 'w')
    f.write('predsa\n')
    for i in predarray_sa:
        f.write(str(i)+'\n')
    f.write('truesa\n')
    for i in truearray_sa:
        f.write(str(i)+'\n')
    f.write('predfl\n')
    for i in predarray_fl:
        f.write(str(i)+'\n')
    f.write('truefl\n')
    for i in truearray_fl:
        f.write(str(i)+'\n')
    f.close()

def plotSolvAccFlex(log_path): #Plots RSA and B-factor predictions
    log_path=log_path
    with open(log_path+'/multitask_log.txt', 'r') as file:
        data = file.read().split('\n')
        predsa=data[1:data.index('truesa')]
        predsa = list(map(float, predsa))
        truesa = data[data.index('truesa') + 1:data.index('predfl')]
        truesa = list(map(float, truesa))
        predfl = data[data.index('predfl') + 1:data.index('truefl')]
        predfl = list(map(float, predfl))
        truefl = data[data.index('truefl') + 1:-1]
        truefl = list(map(float, truefl))

    sns.set()
    sns.set_style('white')
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.4, left=None, right=None)

    sns.lineplot(np.arange(len(truesa)), truesa, label='True', color=sns.color_palette('Paired')[0])
    sns.lineplot(np.arange(len(predsa)), predsa,  label='Pred', color=sns.color_palette('Paired')[1])

    plt.legend(loc='upper left', fontsize='large')
    plt.xlabel('RSA Values')
    axes = plt.gca()
    axes.set_ylim([-0.2, 1.2])
    axes.set_xlim([0, len(predsa)])

    plt.subplot(2, 1, 2)
    plt.tick_params(axis='x', top=False)
    plt.tick_params(axis='y', right=False)

    sns.lineplot(np.arange(len(truefl)), truefl, label='True', color=sns.color_palette('Paired')[0])
    sns.lineplot(np.arange(len(predfl)), predfl,  label='Pred', color=sns.color_palette('Paired')[1])

    plt.legend(loc='upper left', fontsize='large')
    plt.xlabel('B-Factors')
    axes = plt.gca()
    axes.set_ylim([-3, 8])
    axes.set_xlim([0, len(predfl)])
    plt.savefig(log_path + '/CheckSolvAccFlexSampleLinReg.pdf')

def write_predictedData(log_path, dict): #Logs predicted data for later analysis
    np.save(log_path + '/predictedData', dict)


