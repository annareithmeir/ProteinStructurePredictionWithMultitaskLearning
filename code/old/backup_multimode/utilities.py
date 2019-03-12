import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    plt.switch_backend('agg')  # GPU is only available via SSH (no display)
import seaborn as sns

def create_logdir(log_path):
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

def writeLogFile(log_path, lr, dssp_mode, model):
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

def addProgressToLogFile(log_path, epoch, train_loss, test_loss, train_acc, test_acc):
    f=open(log_path+'/log.txt', 'a')
    f.write(str(str(epoch)+' | '+str(train_loss)+', '+str(test_loss)+' | '+str(train_acc)+', '+str(test_acc)))
    f.write('\n')
    f.close()

def writeElapsedTime(log_path, time):
    f = open(log_path + '/log.txt', 'a')
    f.write(str('TOTAL ELAPSED TIME: '+ str(round(time,3))))
    f.close()

def writeNumberOfParameters(log_path, net, param):
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    f=open(log_path, 'a')
    f.write(str(net+' : '+str(param)))
    f.write('\n')
    f.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_bootstrapping(log_path, matrices):  # matrices=list of confusion matrices
    accs = []
    for i in range(len(matrices)):
        acc = matrices[i].trace() / float(matrices[i].sum())
        accs.append(round(acc, 3))

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
    struct3_losses_test=[]
    struct8_losses_test = []
    struct3_losses_train=[]
    struct8_losses_train = []
    solvAcc_losses_test=[]
    solvAcc_losses_train=[]
    flex_losses_test = []
    flex_losses_train = []
    total_losses_test=[]
    total_losses_train=[]

    for l in train_losses:
        struct3_losses_train.append(l[0])
        #struct8_losses_train.append(l[1])
        solvAcc_losses_train.append(l[1])
        flex_losses_train.append(l[2])
        total_losses_train.append(l[3])

    for k in test_losses:
        struct3_losses_test.append(k[0])
        #struct8_losses_test.append(k[1])
        solvAcc_losses_test.append(k[1])
        flex_losses_test.append(k[2])
        total_losses_test.append(k[3])

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 4, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Losses over time')

    struct3_train=sns.lineplot(np.arange(epochs), struct3_losses_train, label='Train', ax=axes[0])
    struct3_test=sns.lineplot(np.arange(epochs), struct3_losses_test, label='Test', ax=axes[0])

    #struct8_train = sns.lineplot(np.arange(epochs), struct8_losses_train, label='Train', ax=axes[1])
    #struct8_test = sns.lineplot(np.arange(epochs), struct8_losses_test, label='Test', ax=axes[1])

    solv_train=sns.lineplot(np.arange(epochs), solvAcc_losses_train, label='Train', ax=axes[1])
    solv_test=sns.lineplot(np.arange(epochs), solvAcc_losses_test, label='Test', ax=axes[1])

    flex_train=sns.lineplot(np.arange(epochs), flex_losses_train, label='Train', ax=axes[2])
    flex_test=sns.lineplot(np.arange(epochs), flex_losses_test, label='Test', ax=axes[2])

    total_train = sns.lineplot(np.arange(epochs), total_losses_train, label='Train', ax=axes[3])
    total_test = sns.lineplot(np.arange(epochs), total_losses_test, label='Test', ax=axes[3])

    struct3_train.set(xlabel=' Structure 3 prediction')
    #struct8_train.set(xlabel=' Structure 3 prediction')
    solv_train.set(xlabel='Solvent accessibility prediction')
    flex_train.set(xlabel='Flexibility prediction')
    total_train.set(xlabel='Multitask prediction')
    struct3_train.set(ylabel='Loss')
    solv_train.set(ylabel=' ')
    flex_train.set(ylabel=' ')
    total_train.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/train_test_loss_multitarget.pdf')


def get_other_scores(conf_mat):
                diagonal_sum = conf_mat.trace()
                sum_of_all_elements = conf_mat.sum()
                print('\n')
                print('ACCURACY: ', round(diagonal_sum / float(sum_of_all_elements),2))
                for label in range(3):
                    col = conf_mat[:, label]
                    row = conf_mat[label, :]
                    precision=conf_mat[label, label] / float(col.sum())
                    recall=conf_mat[label, label] / float(row.sum())
                    f1=2*(recall*precision)/(recall+precision)
                    print('PRECISION CLASS', str(label), '---->', round(precision,2))
                    print('RECALL CLASS: ', str(label), '---->', round(recall,2))
                    #F1 Score = 2*(Recall * Precision) / (Recall + Precision)
                    print('f! SCORE: ', str(label), '---->', round(f1,2))


def get_other_scores_multitask(conf_mat):
    conf_mat = conf_mat[0]
    diagonal_sum = conf_mat.trace()
    sum_of_all_elements = conf_mat.sum()
    print('\n')
    print('ACCURACY: ', round(diagonal_sum / float(sum_of_all_elements), 2))
    for label in range(conf_mat.shape[0]):
        col = conf_mat[:, label]
        print('PRECISION CLASS', str(label), '---->', round(conf_mat[label, label] / float(col.sum()), 2))
        row = conf_mat[label, :]
        print('RECALL CLASS: ', str(label), '---->', round(conf_mat[label, label] / float(row.sum()), 2))
    #f1 score

def plot_accuracy(log_path, train_acc, test_acc, epochs):
    plt.clf()
    plt.figure()
    plt.title('Accuracy')
    plt.plot(np.arange(epochs), test_acc, label='Test')
    plt.plot(np.arange(epochs), train_acc, label='Train')
    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + '/train_test_structure_acc.pdf')

def plot_confmat(log_path, confmat):
    plt.clf()
    plt.figure()

    fig=sns.heatmap(confmat, annot=True, fmt='d')

    conf_mat_cols = 'Predicted Labels'
    conf_mat_rows = 'True labels'
    fig.text(0, 0.96, conf_mat_cols, ha='center', va='center', size=13)
    fig.text(0.06, 0, conf_mat_rows, ha='center', va='center',
             rotation='vertical', size=13)
    plt.savefig(log_path + '/structure_confusion_matrix.pdf')

def plotComparisonForInputType(log_path, input_type, multi_mode):
    dirs=[]
    dictionary=dict()

    for (dirpath, dirnames, filenames) in os.walk(log_path):
        for name in dirnames:
            splits = name.split('_')
            if (splits[1] == input_type):
                dirs.append(str(log_path + name))
        break

    print(dirs)
    #dirs = [s for s in dirs if task_type.lower() in s]
    #dirs=(s for s in dirs if input_type.lower() in s.lower())

    for dir in dirs:
        print('------>', dir)
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-1]
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
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Accuracy comparison all nets')

    for dir in dictionary:
        train_acc=dictionary[dir][2]
        test_acc=dictionary[dir][3]

        plot_train = sns.lineplot(np.arange(len(train_acc)), train_acc, label=dir, ax=axes[0])
        plot_test = sns.lineplot(np.arange(len(test_acc)), test_acc, label=dir, ax=axes[1])
        plot_train.set(xlabel=' Train Accuracy')
        plot_test.set(xlabel='Test Accuracy')
        plot_train.set(ylabel='Accuracy')
        plot_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + 'acc_comparison_'+input_type+'_'+multi_mode+'.pdf')
    plt.close()

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Loss comparison all nets')

    for dir in dictionary:
        train_loss = dictionary[dir][0]
        test_loss = dictionary[dir][1]
        loss_train = sns.lineplot(np.arange(len(train_loss)), train_loss, label=dir, ax=axes[0])
        loss_test = sns.lineplot(np.arange(len(test_loss)), test_loss, label=dir, ax=axes[1])

        loss_train.set(xlabel=' Train Loss')
        loss_test.set(xlabel='Test Loss')
        loss_train.set(ylabel='Loss')
        loss_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    #plt.savefig(log_path + '/loss_comparison_'+input_type+'-'+multi_mode+'.pdf')
    plt.close()

def plotComparisonForNetworkType(log_path, network_type):
    dirs=[]
    dictionary=dict()

    for (dirpath, dirnames, filenames) in os.walk(log_path):
        for name in dirnames:
            splits = name.split('_')
            if (splits[0] == network_type):
                dirs.append(str(log_path + name))
        break

    print(dirs)
    '''
    print('anfang len:', len(dirs))
    if(network_type=='LSTM'):
        dirs = [s for s in dirs if ("LSTM" in s and not "biLSTM" in s)]
    elif (network_type == 'CNN'):
        dirs = [s for s in dirs if ("CNN" in s and not "DenseCNN" in s)]
    elif (network_type == 'Hybrid'):
        dirs = [s for s in dirs if ("Hybrid" in s and not "DenseHybrid" in s)]
    else:
        dirs = (s for s in dirs if network_type.lower() in s.lower())
    '''

    #dirs = [s for s in dirs if task_type.lower() in s]

    for dir in dirs:
        print('------>',dir)
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-1]
            print(len(data))
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
            print(len(train_acc))
            test_acc = list(map(float, test_acc))
            train_loss = list(map(float, train_loss))
            test_loss = list(map(float, test_loss))
            assert ( len(train_acc)== len(test_acc)==len(train_loss)== len(test_loss))
            tmp={dir : [train_loss, test_loss, train_acc, test_acc]}
            dictionary.update(tmp)

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title(str('Accuracy comparison all inputs for '+network_type))

    print('dict len:', len(dictionary))

    for dir in dictionary:
        train_acc=dictionary[dir][2]
        test_acc=dictionary[dir][3]

        plot_train = sns.lineplot(np.arange(len(train_acc)), train_acc, label=dir, ax=axes[0])
        plot_test = sns.lineplot(np.arange(len(test_acc)), test_acc, label=dir, ax=axes[1])
        plot_train.set(xlabel=' Train Accuracy')
        plot_test.set(xlabel='Test Accuracy')
        plot_train.set(ylabel='Accuracy')
        plot_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + 'acc_comparison_'+network_type+'.pdf')
    plt.close()

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Loss comparison all nets')

    for dir in dictionary:
        train_loss = dictionary[dir][0]
        test_loss = dictionary[dir][1]
        loss_train = sns.lineplot(np.arange(len(train_loss)), train_loss, label=dir, ax=axes[0])
        loss_test = sns.lineplot(np.arange(len(test_loss)), test_loss, label=dir, ax=axes[1])

        loss_train.set(xlabel=' Train Loss')
        loss_test.set(xlabel='Test Loss')
        loss_train.set(ylabel='Loss')
        loss_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    #plt.savefig(log_path + '/loss_comparison_'+network_type+'.pdf')
    plt.close()

def plotComparisonForMultitargetType(log_path, network_type, input_type, target_types):
    dirs=[]
    multimodes=['multi3', 'multi2', 'structure']
    dictionary=dict()

    for tt in target_types:
        for mt in multimodes:
            if(os.path.exists(str(log_path+mt+'/'+str(tt)+'/'))):
                for (dirpath, dirnames, filenames) in os.walk(str(log_path+mt+'/'+str(tt)+'/')):
                    for name in dirnames:
                        splits=name.split('_')
                        if(splits[0]==network_type and splits[1] in input_type):
                            dirs.append(str(log_path+mt+'/'+str(tt)+'/'+name))

    print('####',dirs)


    for dir in dirs:
        print('------>',dir)
        train_acc=[]
        test_acc=[]
        train_loss=[]
        test_loss=[]
        with open(dir+'/log.txt', 'r') as file:
            data = file.read().split('\n')
            data=data[4:-1]
            print(len(data))
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
            print(len(train_acc))
            test_acc = list(map(float, test_acc))
            train_loss = list(map(float, train_loss))
            test_loss = list(map(float, test_loss))
            assert ( len(train_acc)== len(test_acc)==len(train_loss)== len(test_loss))
            tmp={dir : [train_loss, test_loss, train_acc, test_acc]}
            dictionary.update(tmp)

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title(str('Accuracy comparison all multitarget modes for '+network_type))

    print('dict len:', len(dictionary))

    for dir in dictionary:
        train_acc=dictionary[dir][2]
        test_acc=dictionary[dir][3]

        plot_train = sns.lineplot(np.arange(len(train_acc)), train_acc, label=dir, ax=axes[0])
        plot_test = sns.lineplot(np.arange(len(test_acc)), test_acc, label=dir, ax=axes[1])
        plot_train.set(xlabel=' Train Accuracy')
        plot_test.set(xlabel='Test Accuracy')
        plot_train.set(ylabel='Accuracy')
        plot_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(log_path + 'acc_comparison_'+network_type+'_multimodes.pdf')
    plt.close()

    plt.clf()
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), sharex=False, sharey=True)
    plt.title('Loss comparison all nets')

    for dir in dictionary:
        train_loss = dictionary[dir][0]
        test_loss = dictionary[dir][1]
        loss_train = sns.lineplot(np.arange(len(train_loss)), train_loss, label=dir, ax=axes[0])
        loss_test = sns.lineplot(np.arange(len(test_loss)), test_loss, label=dir, ax=axes[1])

        loss_train.set(xlabel=' Train Loss')
        loss_test.set(xlabel='Test Loss')
        loss_train.set(ylabel='Loss')
        loss_test.set(ylabel=' ')

    plt.legend(loc=2, fontsize='small')
    #plt.savefig(log_path + '/loss_comparison_'+network_type+'_multimodes.pdf')
    plt.close()
