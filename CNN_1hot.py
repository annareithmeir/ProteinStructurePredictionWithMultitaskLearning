import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

proteins_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/targets/dssp'
seq_path='../mheinzinger/deepppi1tb/contact_prediction/contact_prediction_v2/sequences'

inputMatrix=np.load('matrix_1hot_3_train.npy')
print(inputMatrix.shape)

targets=np.load('targets_train.npy')
print(targets.shape)
