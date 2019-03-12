import os
import shutil
import numpy as np

'''
i=0
j=0
k=0
l=0
all=0
files=[]
for (dirpath, dirnames, filenames) in os.walk('preprocessing'):
    files.extend(dirnames)
    break

for f in files:

    #fsplit=f.split('_')
    #p=fsplit[0]
    #name=fsplit[1]

    if(os.path.exists('/home/mheinzinger/contact_prediction_v2/alignments/'+f+'/'+f+'.psicov') and
            os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/dssp/' + f.lower() + '/' + f.lower() + '.8.consensus.dssp') and
            os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/dssp/' + f.lower() + '/' + f.lower() + '.3.consensus.dssp')
    ):
        all+=1


    if(os.path.exists('preprocessing/' + f + '/protvec+scoringmatrix.npy') and
            os.path.exists('preprocessing/' + f + '/protvec.npy') and
            os.path.exists('preprocessing/' + f + '/1hot.npy') and
            os.path.exists('preprocessing/' + f + '/protvecevolutionary.npy')
    ):
        i+=1

    if (os.path.exists('preprocessing/' + f + '/protvec+scoringmatrix.npy') and
            os.path.exists('preprocessing/' + f + '/protvec.npy') and
            os.path.exists('preprocessing/' + f + '/1hot.npy') and
            os.path.exists('preprocessing/' + f + '/protvecevolutionary.npy') and
            os.path.exists('preprocessing/' + f + '/structures_8.npy') and
            os.path.exists('preprocessing/' + f + '/mask_8.npy') and
            os.path.exists('preprocessing/' + f + '/structures_3.npy') and
            os.path.exists('preprocessing/' + f + '/mask_3.npy') and
            os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/bdb_bvals/' + f.lower() + '.bdb.memmap') and
            os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/dssp/' + f.lower() + '/' + f.lower() + '.rel_asa.memmap')

    ):
        j+=1

    if (os.path.exists('preprocessing/' + f + '/mask_3.npy') and
            os.path.exists('preprocessing/' + f + '/mask_8.npy') and
            os.path.exists('preprocessing/' + f + '/structures_3.npy') and
            os.path.exists('preprocessing/' + f + '/structures_8.npy') ):
        k += 1

    if(os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/bdb_bvals/' + f.lower() + '.bdb.memmap') and
    os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/dssp/' + f.lower() + '/' + f.lower() + '.rel_asa.memmap')):
        l+=1

print(all)
print('---')
print('all inputs there:',i)
print('everything there:',j)
print('all struct there:',k)
print('RSA and Bfac there:',l)
'''

def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

files=[]
for (dirpath, dirnames, filenames) in os.walk('preprocessing'):
    files.extend(dirnames)
    break

for f in files:
    if (os.path.exists('preprocessing/' + f + '/protvec+scoringmatrix.npy') and
                os.path.exists('preprocessing/' + f + '/protvec.npy') and
                os.path.exists('preprocessing/' + f + '/1hot.npy') and
                os.path.exists('preprocessing/' + f + '/protvecevolutionary.npy') and
                os.path.exists('preprocessing/' + f + '/structures_8.npy') and
                os.path.exists('preprocessing/' + f + '/mask_8.npy') and
                os.path.exists('preprocessing/' + f + '/structures_3.npy') and
                os.path.exists('preprocessing/' + f + '/mask_3.npy') and
                os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/bdb_bvals/' + f.lower() + '.bdb.memmap') and
                os.path.exists('/home/mheinzinger/contact_prediction_v2/targets/dssp/' + f.lower() + '/' + f.lower() + '.rel_asa.memmap')

        ):
            copyDirectory('preprocessing/' + f, 'dataset_preprocessed/'+f)






