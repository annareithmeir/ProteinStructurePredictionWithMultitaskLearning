import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utilities
sns.set_palette('Paired')

#
# This plots miscellaneous plots not related to the code, i.e.
# 1. The pdb/swissprot plot
#



#Create plot of PDB/SwissProt entries
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(style="ticks", context="poster")

pdb=pd.read_csv('/home/anna/Dokumente/Bachelorarbeit/tmp_data/PDBGrowth.csv')
swiss = {'Year': [1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,
                  1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000, 2001, 2002, 2003, 2004,
                  2005, 2006, 2007, 2008, 2009, 2010, 2011,2012,2013, 2014, 2015, 2016, 2017, 2018, 2019],
     'Total Number of Entries Available': [0,0,0,0,0,0,0,0,0,0,
                                           4160, 5205, 8702, 12305, 18364, 22654, 28154, 33329, 40292, 49340, 59021, 69113, 77977, 80000, 86593, 101602, 122564, 135850, 163235,
                                           194317, 241242, 276256, 392667, 428650, 523151, 532146, 538585, 541762, 547085, 550116, 553231, 556568, 558590, 559228]}

sns.lineplot(pdb['Year'], pdb['Total Number of Entries Available'], label='PDB entries')
sns.lineplot(swiss['Year'], swiss['Total Number of Entries Available'], label='UniProtKB/Swiss-Prot entries')
plt.ylim([0,600000])
ax=plt.gca()
ax.yaxis.set_major_formatter(ticker.EngFormatter())
plt.ylabel('Entries')
plt.savefig('/home/anna/Dokumente/Bachelorarbeit/images/PDBGrowth.pdf')






