import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os
from matplotlib import rcParams

#
# This file plots the results of the data analysis of the predictions, saved in 'data analysis/predicted'.
# The plots are also saved there.
#

plot_path='data analysis/predicted'
sns.set(style="white", palette="muted", color_codes=True)
sns.set_style('ticks')
sns.set_context('talk')
sns.despine()

classCounts3=np.load(plot_path+'/countClasses_3.npy')
classCounts8=np.load(plot_path+'/countClasses_8.npy')
classCounts8=np.nan_to_num(classCounts8)


#
# PLOT PIECHARTS
#
classCounts8=[classCounts8[4], classCounts8[7],classCounts8[3],classCounts8[2],classCounts8[0],classCounts8[5],classCounts8[1],classCounts8[6], classCounts8[8]]
labels8=np.array(['T','-','S','H','I','G','B','E','X or Y'])
colors8=np.array(['darksalmon', 'indianred', 'lightcoral','forestgreen', 'yellowgreen','palegreen' ,'powderblue','lightskyblue','gainsboro'])
explode8=np.array([0,0,0,0,0,0,0,0,0.1])
classCounts8=np.array(classCounts8)
mask=np.where(classCounts8>0)
labels8=labels8[mask]
colors8=colors8[mask]
explode8=explode8[mask]
classCounts8=classCounts8[mask]

fig1, ax1 = plt.subplots()
ax1.pie(classCounts3, labels=['C', 'H', 'E', 'X or Y'], autopct='%1.1f%%',pctdistance=0.7,colors=['lightcoral', 'yellowgreen', 'powderblue', 'gainsboro'], shadow=False, startangle=90, explode=[0,0,0,0.1], textprops={'fontsize': 25})
ax1.axis('equal')
plt.tight_layout()
fig1.savefig(plot_path+'/piechart_3classes_pred.pdf')

plt.clf()
fig1, ax1 = plt.subplots()
ax1.pie(classCounts8, labels=labels8, autopct='%1.1f%%',pctdistance=0.7,colors=colors8, shadow=False, explode=explode8,startangle=90, textprops={'fontsize': 25})
ax1.axis('equal')
plt.tight_layout()
plt.savefig(plot_path+'/piechart_8classes_pred.pdf')

#
# PLOT CHAIN LENGTHS 3
#
chainCounts3=pd.read_pickle(plot_path+'/countChains3')
plt.clf()
fig, axes = plt.subplots(1, 3, figsize=(28,14), sharex=False, sharey=True)
sns.set_context('poster')

first=sns.barplot(np.arange(25),chainCounts3['C'][:25] ,color='salmon',  ax=axes[0])
first.tick_params(axis='x', top=False)
first.tick_params(axis='y', right=False)
first.axvline(4.8,linestyle='dashed', color='grey')
second=sns.barplot(np.arange(40),chainCounts3['H'][:40] ,color='yellowgreen',  ax=axes[ 1])
second.tick_params(axis='x', top=False)
second.tick_params(axis='y', right=False)
second.axvline(10.4,linestyle='dashed', color='grey')
third=sns.barplot(np.arange(20),chainCounts3['E'][:20] ,color='cornflowerblue',  ax=axes[ 2])
third.tick_params(axis='x', top=False)
third.tick_params(axis='y', right=False)
third.axvline(5.5,linestyle='dashed', color='grey')
first.set(xticks=[0,5,10,15,20,25])
first.set(xticklabels=[0,5,10,15,20,25])
second.set(xticks=[0,10,20,30,40])
second.set(xticklabels=[0,10,20,30,40])
third.set(xticks=[0,5,10,15,20])
third.set(xticklabels=[0,5,10,15,20])

first.set_xlabel(r'$L_C$', fontsize=30)
second.set_xlabel(r'$L_H$',fontsize=30)
third.set_xlabel(r'$L_E$',fontsize=30)
first.set_ylabel('Occurences',fontsize=30)
second.set(ylabel=' ')
third.set(ylabel=' ')


plt.savefig(plot_path+'/chains3_pred.pdf')
