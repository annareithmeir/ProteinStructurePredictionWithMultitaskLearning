import numpy as np
import os
import shutil

#
# This file splits the dataset into static train, test and validation sets
# Ratios used:
#   Test set: 20\%
#   Rest:  Train set: 80%, Val set: 20%
#


def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

#Create testset
files=[]
for (dirpath, dirnames, filenames) in os.walk('trainset_preprocessed'):
    files.extend(dirnames)
    break


for i in range(len(files)):
    rnd = np.random.rand()
    if(rnd<0.2):
        print(files[i])
        copyDirectory('trainset_preprocessed/'+files[i],'testset_preprocessed/'+files[i])
        shutil.rmtree('trainset_preprocessed/'+files[i])


#Create validationset from remaining samples
files=[]
for (dirpath, dirnames, filenames) in os.walk('trainset_preprocessed'):
    files.extend(dirnames)
    break


for i in range(len(files)):
    rnd = np.random.rand()
    if(rnd<0.2):
        print(files[i])
        copyDirectory('trainset_preprocessed/'+files[i],'validationset_preprocessed/'+files[i])
        shutil.rmtree('trainset_preprocessed/'+files[i])


