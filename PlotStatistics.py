import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

plot_path='stats'
sns.set(style="white", palette="muted", color_codes=True)
sns.set_style('ticks')
sns.set_context('paper') #'paper'
sns.despine()

dict=np.load(plot_path+'/stats_dict.npy').item() #avg3, avg8

classCounts3=np.load(plot_path+'/countClasses_3.npy')
classCounts8=np.load(plot_path+'/countClasses_8.npy')
# array [c,h,e,xy] of occurences

plt.figure()

#
# PLOT PIECHARTS
#

classcounts8_ordered=[classCounts8[4], classCounts8[3],classCounts8[7],classCounts8[0],classCounts8[2],classCounts8[5],classCounts8[6],classCounts8[1], classCounts8[8]]

plt.pie(classCounts3, labels=['C', 'H', 'E', 'X or Y'], autopct='%1.1f%%',pctdistance=1.4,colors=['gold', 'yellowgreen', 'lightcoral', 'grey'], shadow=False, startangle=90, explode=[0,0,0,0.1])
plt.savefig(plot_path+'/piechart_3classes.pdf')
plt.clf()
plt.pie(classcounts8_ordered, labels=['T','S','-','H','I','G','B','E','X or Y'], autopct='%1.1f%%',pctdistance=1.4,colors=['gold', 'orange', 'yellow', 'yellowgreen','palegreen', 'forestgreen','lightcoral','firebrick','grey'], shadow=False, explode=[0,0,0,0,0,0,0,0,0.1],startangle=90)
plt.savefig(plot_path+'/piechart_8classes.pdf')

#
# PLOT LENGTHS
#
lengthCounts3=np.load(plot_path+'/lengths3.npy')
print('AVERAGE LENGTH---->',np.average(lengthCounts3))

sns.set_context('talk')
plt.clf()
dist3plot=sns.distplot(lengthCounts3, kde=False)
plt.title('Occurences of sequence lengths in data set')
dist3plot.set(xlabel='Sequence length')
dist3plot.set(ylabel='Occurences')
#dist3plot.set(xlim=[0, 1200])
plt.axvline(np.average(lengthCounts3),linestyle='dashed')
fig=dist3plot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/lengths.pdf')


#
# PLOT CHAIN LENGTHS 3
#
chainCounts3=pd.read_pickle(plot_path+'/countChains3')
#columns=c,h,e, rows=lengths


plt.clf()
fig, axes = plt.subplots(1, 3, figsize=(28,14), sharex=False, sharey=True)

first=sns.barplot(np.arange(50),chainCounts3['C'][:50] , color="skyblue", ax=axes[0])
second=sns.barplot(np.arange(50),chainCounts3['H'][:50] , color="olive", ax=axes[ 1])
third=sns.barplot(np.arange(50),chainCounts3['E'][:50] , color="green", ax=axes[ 2])
first.set(xticks=[10,20,30,40,50])
first.set(xticklabels=[10,20,30,40,50])
second.set(xticks=[10,20,30,40,50])
second.set(xticklabels=[10,20,30,40,50])
third.set(xticks=[10,20,30,40,50])
third.set(xticklabels=[10,20,30,40,50])

first.set(xlabel=' Chain length C')
second.set(xlabel='Chain length H')
third.set(xlabel='Chain length E')
first.set(ylabel='Occurences')
second.set(ylabel=' ')
third.set(ylabel=' ')


plt.savefig(plot_path+'/chains3.pdf')

#
# PLOT CHAIN LENGTHS 8
#
chainCounts8=pd.read_pickle(plot_path+'/countChains8')
print(chainCounts8.max())
plt.clf()
first=sns.barplot(np.arange(50),chainCounts8['H'][:50], color="skyblue")
first.set(xticks=[10,20,30,40,50])
first.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_H.pdf')

plt.clf()
second=sns.barplot(np.arange(50),chainCounts8['E'][:50], color="olive")
second.set(xticks=[10,20,30,40,50])
second.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_E.pdf')

plt.clf()
third=sns.barplot(np.arange(50),chainCounts8['I'][:50], color="skyblue")
third.set(xticks=[10,20,30,40,50])
third.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_I.pdf')

plt.clf()
fourth=sns.barplot(np.arange(50),chainCounts8['S'][:50], color="skyblue")
fourth.set(xticks=[10,20,30,40,50])
fourth.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_S.pdf')

plt.clf()
fifth=sns.barplot(np.arange(50),chainCounts8['T'][:50], color="skyblue")
fifth.set(xticks=[10,20,30,40,50])
fifth.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_T.pdf')

plt.clf()
sixth=sns.barplot(np.arange(50),chainCounts8['G'][:50], color="skyblue")
sixth.set(xticks=[10,20,30,40,50])
sixth.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_G.pdf')

plt.clf()
seventh=sns.barplot(np.arange(50),chainCounts8['B'][:50], color="skyblue")
seventh.set(xticks=[10,20,30,40,50])
seventh.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_B.pdf')

plt.clf()
eightth=sns.barplot(np.arange(50),chainCounts8['-'][:50], color="skyblue")
eightth.set(xticks=[10,20,30,40,50])
eightth.set(xticklabels=[10,20,30,40,50])
plt.savefig(plot_path+'/chains8_none.pdf')


plt.savefig(plot_path+'/chains8.pdf')

def together():
    chainCounts8=pd.read_pickle(plot_path+'/countChains8')
    print(chainCounts8.max())
    plt.clf()
    fig, axes = plt.subplots(2, 4, figsize=(28,14), sharex=True, sharey=True)
    first=sns.barplot(np.arange(50),chainCounts8['H'][:50], color="skyblue", ax=axes[0,0])
    #plt.savefig(plot_path+'/chains8_H.pdf')

    #plt.clf()
    second=sns.barplot(np.arange(50),chainCounts8['E'][:50], color="olive", ax=axes[0,1])
    #plt.savefig(plot_path+'/chains8_E.pdf')

    #plt.clf()
    third=sns.barplot(np.arange(50),chainCounts8['I'][:50], color="skyblue", ax=axes[0,2])
    #plt.savefig(plot_path+'/chains8_I.pdf')

    #plt.clf()
    fourth=sns.barplot(np.arange(50),chainCounts8['S'][:50], color="skyblue", ax=axes[0,3])
    #plt.savefig(plot_path+'/chains8_S.pdf')

    #plt.clf()
    fifth=sns.barplot(np.arange(50),chainCounts8['T'][:50], color="skyblue", ax=axes[1,0])
    #plt.savefig(plot_path+'/chains8_T.pdf')

    #plt.clf()
    sixth=sns.barplot(np.arange(50),chainCounts8['G'][:50], color="skyblue", ax=axes[1,1])
    #plt.savefig(plot_path+'/chains8_G.pdf')

    #plt.clf()
    seventh=sns.barplot(np.arange(50),chainCounts8['B'][:50], color="skyblue", ax=axes[1,2])
    #plt.savefig(plot_path+'/chains8_B.pdf')

    #plt.clf()
    eightth=sns.barplot(np.arange(50),chainCounts8['-'][:50], color="skyblue", ax=axes[1,3])
    #plt.savefig(plot_path+'/chains8_none.pdf')
    fifth.set(xticks=[10,20,30,40,50])
    sixth.set(xticks=[10,20,30,40,50])
    seventh.set(xticks=[10,20,30,40,50])
    eightth.set(xticks=[10,20,30,40,50])

    first.set(xlabel='H')
    second.set(xlabel='E')
    third.set(xlabel='I')
    fourth.set(xlabel='S')
    fifth.set(xlabel='T')
    sixth.set(xlabel='G')
    seventh.set(xlabel='B')
    eightth.set(xlabel='-')
    first.set(ylabel='Occurences')
    fifth.set(ylabel='Occurences')
    second.set(ylabel=' ')
    third.set(ylabel=' ')
    fourth.set(ylabel=' ')
    sixth.set(ylabel=' ')
    seventh.set(ylabel=' ')
    eightth.set(ylabel=' ')

    fig.suptitle('Chain lengths of DSSP8 classes')

    plt.savefig(plot_path+'/chains8.pdf')

#countAAs3=np.load(plot_path+'/countAAs3.npy')
#countAAs8=np.load(plot_path+'/countAAs8.npy')


solvAcc_avgs=np.load(plot_path+'/solvAcc_avgs.npy')
solvAcc_quarters=np.load(plot_path+'/solvAcc_quarters.npy')
solvAcc_halfs=np.load(plot_path+'/solvAcc_halfs.npy')
print(solvAcc_halfs)

sns.set_context('paper')
plt.clf()
solvAccAvgsPlot=sns.distplot(solvAcc_avgs, kde=False)
plt.title('Averages of Solvent Accessibilities per sequence')
solvAccAvgsPlot.set(xlabel='Average')
solvAccAvgsPlot.set(ylabel='Occurences')
#dist3plot.set(xlim=[0, 1200])
plt.axvline(np.average(solvAcc_avgs),linestyle='dashed')
fig=solvAccAvgsPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/solvAcc_avgs.pdf')

sns.set_context('paper')
plt.clf()
solvAccQuartersPlot=sns.barplot(np.arange(4),solvAcc_quarters)
plt.title('Quarters of solv Acc occurences')
solvAccQuartersPlot.set(xlabel='Quarter')
solvAccQuartersPlot.set(ylabel='Occurences')
fig=solvAccQuartersPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/solvAcc_quarters.pdf')

sns.set_context('paper')
plt.clf()
df=pd.DataFrame({'state': ['Buried', 'Exposed'], 'occs': [solvAcc_halfs[0], solvAcc_halfs[1]]})
sum=solvAcc_halfs[0]+solvAcc_halfs[1]
solvAccHalfsPlot=sns.barplot(x='state',y='occs', data=df)
plt.title('Buried or Exposed Occurences - Buried: '+str(solvAcc_halfs[0]/float(sum))+' Exposed: '+str(solvAcc_halfs[1]/float(sum)))
solvAccHalfsPlot.set(xlabel='Buried or exposed')
solvAccHalfsPlot.set(ylabel='Occurences')
fig=solvAccHalfsPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/solvAcc_halfs.pdf')


flex_avgs=np.load(plot_path+'/flex_avgs.npy')
flex_thirds=np.load(plot_path+'/flex_thirds.npy')

sns.set_context('paper')
plt.clf()
flexAvgsPlot=sns.distplot(flex_avgs, kde=False)
plt.title('Averages of flex per sequence')
flexAvgsPlot.set(xlabel='Average')
flexAvgsPlot.set(ylabel='Occurences')
#dist3plot.set(xlim=[0, 1200])
plt.axvline(np.average(flex_avgs),linestyle='dashed')
fig=flexAvgsPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/flex_avgs.pdf')

sns.set_context('paper')
plt.clf()
flexQuartersPlot=sns.barplot(np.arange(3),flex_thirds)
plt.title('Quarters of flex occurences')
flexQuartersPlot.set(xlabel='third')
flexQuartersPlot.set(ylabel='Occurences')
fig=flexQuartersPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/flex_thirds.pdf')
