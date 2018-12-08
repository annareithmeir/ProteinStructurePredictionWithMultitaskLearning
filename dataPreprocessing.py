import numpy as np
import os

'''
for prot in os.listdir('datapreprocessing'):
    if(len(os.listdir('datapreprocessing/'+prot))!=5):
        print(prot)
'''

for prot in os.listdir('proteins'):
    if(len(os.listdir('proteins/'+prot))!=5):
        print(prot)
