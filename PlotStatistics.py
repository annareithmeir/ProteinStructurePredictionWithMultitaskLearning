import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import os

plot_path='stats'
sns.set(style="white", palette="muted", color_codes=True)
sns.set_style('ticks')

dict=np.load(plot_path+'/stats_dict.npy').item() #avg3, avg8

classCounts3=np.load(plot_path+'/countClasses_3.npy')
classCounts8=np.load(plot_path+'/countClasses_8.npy')
# array [c,h,e,xy] of occurences
print(classCounts8)
print(classCounts3)

plt.figure()
plt.pie(classCounts3, labels=['C', 'H', 'E', 'X or Y'], autopct='%1.1f%%',colors=['gold', 'yellowgreen', 'lightcoral', 'grey'], shadow=False, startangle=90, explode=[0,0,0,0.1])
plt.savefig(plot_path+'/piechart_3classes.pdf')
plt.clf()
plt.pie(classCounts8, labels=['H','E','I','S','T','G','B','-','X or Y'], autopct='%1.1f%%',pctdistance=1.3,colors=['gold', 'orange', 'yellow', 'yellowgreen','palegreen', 'forestgreen','lightcoral','firebrick','grey'], shadow=False, explode=[0,0,0,0,0,0,0,0,0.1],startangle=90)
plt.savefig(plot_path+'/piechart_8classes.pdf')

lengthCounts3=np.load(plot_path+'/lengths3.npy')

plt.clf()
dist3plot=sns.distplot(lengthCounts3)
fig=dist3plot.get_figure()
plt.tight_layout()
fig.savefig(plot_path+'/lengths.pdf')



chainCounts3=pd.read_pickle(plot_path+'/countChains3')
chainCounts8=pd.read_pickle(plot_path+'/countChains8')
#columns=c,h,e, rows=lengths


plt.clf()
f, axes = plt.subplots(1, 3, figsize=(14,14), sharex=False, sharey=True)

first=sns.barplot(np.arange(60),chainCounts3['C'][:60] , color="skyblue", ax=axes[0])
second=sns.barplot(np.arange(60),chainCounts3['H'][:60] , color="olive", ax=axes[ 1])
third=sns.barplot(np.arange(60),chainCounts3['E'][:60] , color="green", ax=axes[ 2])
first.set(xticks=[10,20,30,40,50,60])
first.set(xticklabels=[10,20,30,40,50,60])
second.set(xticks=[10,20,30,40,50,60])
second.set(xticklabels=[10,20,30,40,50,60])
third.set(xticks=[10,20,30,40,50,60])
third.set(xticklabels=[10,20,30,40,50,60])

first.set(xlabel='Chain length C')
second.set(xlabel='Chain length H')
third.set(xlabel='Chain length E')
first.set(ylabel='Occurences')
second.set(ylabel=' ')
third.set(ylabel=' ')

plt.savefig(plot_path+'/chains3.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['H'])),chainCounts8['H'], color="skyblue")
plt.savefig(plot_path+'/chains8_H.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['E'])),chainCounts8['E'], color="olive")
plt.savefig(plot_path+'/chains8_E.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['I'])),chainCounts8['I'], color="skyblue")
plt.savefig(plot_path+'/chains8_I.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['S'])),chainCounts8['S'], color="skyblue")
plt.savefig(plot_path+'/chains8_S.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['T'])),chainCounts8['T'], color="skyblue")
plt.savefig(plot_path+'/chains8_T.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['G'])),chainCounts8['G'], color="skyblue")
plt.savefig(plot_path+'/chains8_G.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['B'])),chainCounts8['B'], color="skyblue")
plt.savefig(plot_path+'/chains8_B.pdf')

plt.clf()
plot=sns.barplot(np.arange(len(chainCounts8['-'])),chainCounts8['-'], color="skyblue")
plt.savefig(plot_path+'/chains8_none.pdf')

#countAAs3=np.load(plot_path+'/countAAs3.npy')
#countAAs8=np.load(plot_path+'/countAAs8.npy')