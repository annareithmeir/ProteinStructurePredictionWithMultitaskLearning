import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os
from matplotlib import rcParams

#
# This file plots the data analysis results in the 'data analysis' folder.
# The plots are also saved there.
#

colors = ['powder blue','light sky blue', 'dark salmon', 'indian red', 'coral','pale green', 'yellow green', 'forest green']
pal=sns.xkcd_palette(colors)

plot_path='data analysis'
sns.set(style="white", palette="muted", color_codes=True)
sns.set_style('ticks')
sns.set_context('talk')
sns.despine()

dict=np.load(plot_path+'/stats_dict.npy').item() #avg3, avg8

classCounts3=np.load(plot_path+'/countClasses_3.npy')
classCounts8=np.load(plot_path+'/countClasses_8.npy')

#
# PLOT PIECHARTS
#

classcounts8_ordered=[classCounts8[4], classCounts8[3],classCounts8[7],classCounts8[5],classCounts8[0],classCounts8[2],classCounts8[1],classCounts8[6], classCounts8[8]]

fig1, ax1 = plt.subplots()
ax1.pie(classCounts3, labels=['C', 'H', 'E', 'X or Y'], autopct='%1.1f%%',pctdistance=0.7,colors=['lightcoral', 'yellowgreen', 'powderblue', 'gainsboro'], shadow=False, startangle=90, explode=[0,0,0,0.1], textprops={'fontsize': 25})
ax1.axis('equal')
plt.tight_layout()
fig1.savefig(plot_path+'/piechart_3classes.pdf')

#sns.set_context('talk')
fig1, ax1 = plt.subplots()
ax1.pie(classcounts8_ordered, labels=['T','S','-','G','H','I','E','B','X or Y'],pctdistance=0.7, autopct='%1.1f%%',colors=['darksalmon', 'indianred', 'lightcoral','forestgreen', 'yellowgreen','palegreen' ,'powderblue','lightskyblue','gainsboro'], textprops={'fontsize': 25}, shadow=False, explode=[0,0,0,0,0,0,0,0,0.1],startangle=90)
ax1.axis('equal')
fig1.tight_layout()
fig1.savefig(plot_path+'/piechart_8classes.pdf')

#
# PLOT LENGTHS
#
sns.set()
sns.set_style('white')
sns.set_context('talk')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
lengthCounts3=np.load(plot_path+'/lengths3.npy')
print('AVERAGE LENGTH---->',np.average(lengthCounts3))

dist3plot=sns.distplot(lengthCounts3, kde=True, color=sns.color_palette('Paired')[0])
#plt.title('Occurences of sequence lengths in data set')
dist3plot.set_xlabel('Sequence Lengths', fontsize=25)
dist3plot.set_ylabel('Occurences', fontsize=25)
#dist3plot.set(xlim=[0, 1200])
plt.axvline(np.average(lengthCounts3),linestyle='dashed', color=sns.color_palette('Paired')[1])
fig=dist3plot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/lengths.pdf')


#
# PLOT CHAIN LENGTHS 3
#
chainCounts3=pd.read_pickle(plot_path+'/countChains3')
#columns=c,h,e, rows=lengths

sns.set_context('poster')
plt.clf()
sns.set_context('poster')
fig, axes = plt.subplots(1, 3, figsize=(28,14), sharex=False, sharey=True)
sns.set_context('poster')


first=sns.barplot(np.arange(25),chainCounts3['C'][:25] ,color='salmon',  ax=axes[0])
first.tick_params(axis='x', top=False)
first.tick_params(axis='y', right=False)
first.axvline(4.2,linestyle='dashed', color='grey')
second=sns.barplot(np.arange(40),chainCounts3['H'][:40] ,color='yellowgreen',  ax=axes[ 1])
second.tick_params(axis='x', top=False)
second.tick_params(axis='y', right=False)
second.axvline(9.8,linestyle='dashed', color='grey')
third=sns.barplot(np.arange(20),chainCounts3['E'][:20] ,color='cornflowerblue',  ax=axes[ 2])
third.tick_params(axis='x', top=False)
third.tick_params(axis='y', right=False)
third.axvline(4.7,linestyle='dashed', color='grey')
first.set(xticks=[0,5,10,15,20,25])
first.set(xticklabels=[0,5,10,15,20,25])
second.set(xticks=[0,10,20,30,40])
second.set(xticklabels=[0,10,20,30,40])
third.set(xticks=[0,5,10,15,20])
third.set(xticklabels=[0,5,10,15,20])

first.set_xlabel(r'$L_C$', fontsize=30)
second.set_xlabel(r'$L_H$', fontsize=30)
third.set_xlabel(r'$L_E$', fontsize=30)
first.set_ylabel('Occurences', fontsize=30)
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


#
# RSA values
#

solvAcc_avgs=np.load(plot_path+'/solvAcc_avgs.npy')
solvAcc_quarters=np.load(plot_path+'/solvAcc_quarters.npy')
solvAcc_halfs=np.load(plot_path+'/solvAcc_halfs.npy')

sns.set()
sns.set_style('white')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
solvAccAvgsPlot=sns.distplot(solvAcc_avgs, kde=False)
solvAccAvgsPlot.set(xlabel='Average')
solvAccAvgsPlot.set(ylabel='Occurences')
plt.axvline(np.average(solvAcc_avgs),linestyle='dashed')
fig=solvAccAvgsPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/solvAcc_avgs.pdf')

sns.set()
sns.set_style('white')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
solvAccQuartersPlot=sns.barplot(np.arange(4),solvAcc_quarters)
plt.title('Quarters of solv Acc occurences')
solvAccQuartersPlot.set(xlabel='Quarter')
solvAccQuartersPlot.set(ylabel='Occurences')
fig=solvAccQuartersPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/solvAcc_quarters.pdf')

#
# B-factors
#

sns.set()
sns.set_style('white')
#sns.set_context('talk')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
sum=float(solvAcc_halfs[0] + solvAcc_halfs[1])
df=pd.DataFrame({'state': ['Buried', 'Exposed'], 'occs': [solvAcc_halfs[0]/sum, solvAcc_halfs[1]/sum]})
sum=solvAcc_halfs[0]+solvAcc_halfs[1]
solvAccHalfsPlot=sns.barplot(x='state',y='occs', data=df, palette='Paired') #, label='big'
solvAccHalfsPlot.set_ylabel('%', fontsize=25)
solvAccHalfsPlot.set(xlabel='')
fig=solvAccHalfsPlot.get_figure()
plt.tight_layout()
plt.xticks(fontsize=20)
fig.savefig(plot_path+'/solvAcc_halfs.pdf')


flex_avgs=np.load(plot_path+'/flex_avgs.npy')
flex_thirds=np.load(plot_path+'/flex_thirds.npy')
sum=float(flex_thirds[0]+flex_thirds[1]+flex_thirds[2])
flex_thirds=flex_thirds/sum

sns.set()
sns.set_style('white')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
flexAvgsPlot=sns.distplot(flex_avgs, kde=False)
plt.title('Averages of flex per sequence')
flexAvgsPlot.set(xlabel='Average')
flexAvgsPlot.set(ylabel='Occurences')
plt.axvline(np.average(flex_avgs),linestyle='dashed')
fig=flexAvgsPlot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/flex_avgs.pdf')

sns.set()
sns.set_context('talk')
sns.set_style('white')
plt.clf()
plt.figure()
plt.tick_params(axis='x', top=False)
plt.tick_params(axis='y', right=False)
flexQuartersPlot=sns.barplot(['Not Flexible', 'Average Flexible', 'Flexible',],flex_thirds, palette='Paired', label='big')
flexQuartersPlot.set_ylabel('%', fontsize=25)
flexQuartersPlot.set(xlabel='')
fig=flexQuartersPlot.get_figure()
plt.tight_layout()
plt.xticks(fontsize=20)
fig.savefig(plot_path+'/flex_thirds.pdf')
