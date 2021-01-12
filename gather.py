# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:26:05 2020

@author: dmm
"""
import csv
import os
import json
import numpy as np

from glob import glob
ASIA = {}
with open('D:/AI/scripts/labels_DTI.csv','r') as csvf:
    csvReader = csv.reader(csvf)
    for row in csvReader:
        ASIA.update({row[0]:{"L":int(row[1]),"S":int(row[2]),"found":0}})
        
all_folders = glob('D:/AI/AI_DTI_2/Good_enough/*')
no_asia = []
# with open('D:/AI/AI_DTI_2/sub_acc2.csv','w') as csvf:
    
for folder in all_folders:
    acc_no  = os.path.split(folder)[1]
    if (acc_no in ASIA.keys() and len(glob(os.path.join(folder,'*ACTUAL_DTI*/denoise_moco_dwi_mean_seg.nii'))) > 0):
    # if (acc_no in ASIA.keys() and len(glob(os.path.join(folder,'*ACTUAL_DTI*/denoise_moco_dwi_mean_seg.nii'))) > 0):
        #if len(glob(os.path.join(folder,'*ACTUAL_DTI*/denoise_moco_dwi_mean_seg.nii'))) > 0:
        # csvf.write('{}\n'.format(acc_no))
        ASIA[acc_no]["found"] = 1
    else:
        no_asia.append(acc_no)
all_keys = set(ASIA.keys())
for key in all_keys:
    if not ASIA[key]["found"]:
        del ASIA[key]
        
with open('D:/AI/scripts/labels_DTI.json','w') as jsf:
    json.dump(ASIA,jsf)