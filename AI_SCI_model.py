#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:05:31 2020

@author: josh
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import nibabel as nib
import pandas as pd
import os
import traceback
from scipy import ndimage
from pathlib import Path


Final_array = np.zeros((5000,64,64,1))
FA = 1
slice_count=0
AIS_list = []
AIS_list_by_slice = []
FA_list = []
All_data = 'D:/AI/AI_DTI_2/Good_enough'
xlate = {"A":0,"B":1,"C":2,"D":3, "E":5}
# read AIS labels and convert to dictionary
AIS_labels = 'D:/AI/AIS_label.csv'
AIS_df = pd.read_csv(AIS_labels, usecols=['Accession','AIS'])
AI_AIS_dict = AIS_df.set_index('Accession')['AIS'].to_dict()

#get them masks boiiii
for mask_name in glob.glob('D:/AI/AI_DTI_2/Good_enough/*/*ACTUAL_DTI*/denoise_moco_dwi_mean_seg.nii'):
    acc_path = Path(mask_name)
    sub_acc = acc_path.parents[1].name

#take the masks that match the accession keys
#for accession in AI_AIS_dict.keys():
    if sub_acc in AI_AIS_dict.keys():
        mask_img = nib.load(mask_name)
        mask = mask_img.get_fdata()
        V = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2],FA))
#load the DTI data            
        FA_map = glob.glob(os.path.join(All_data, sub_acc, '*ACTUAL_DTI*/*ACTUAL_DTIFA.nii.gz'))[0]
        FA_img = nib.load(FA_map)
        FA_data = FA_img.get_fdata()
        V[:,:,:,0] = FA_data
        

        V[:,:,:,0][mask<1] = 0
        for aa in range(0,FA_data.shape[2]):
            FA_list.append(aa)
#determine center of mass of the mask, crop to 64x64 and apply that to the dti data array                
        for ss in range(0,V.shape[2]):
            com = ndimage.measurements.center_of_mass(V[:,:,ss])
            if not np.isnan(com[0]):
                com0=np.max((com[0],32))
                com0=np.min((com0,FA_data.shape[0]-32))
                com1=np.max((com[1],32))
                com1=np.min((com1,FA_data.shape[1]-32))
                FA_slice = FA_data[:,:,ss]
                FA_slice[FA_data[:,:,ss]<1]=0
                try:
                    Final_array[slice_count,:,:,0] = FA_slice[int(com0)-32:int(com0)+32,int(com1)-32:int(com1)+32]
                except:
                    print(FA_slice.shape)
                    print('{},{}'.format(com0,com1))
                    traceback.print_exc()
            slice_count = slice_count + 1
            AIS_list.append(xlate[AI_AIS_dict[sub_acc]])    


Final_array = Final_array[0:len(AIS_list),:,:,:]   
 
   
ok = tf.keras.utils.to_categorical(AIS_list, dtype='uint8')
model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3),input_shape=(64,64,1),activation='relu',data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'),
        keras.layers.Conv2D(16, (3,3),input_shape=(64,64,1),activation='relu',data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last'),
        keras.layers.Flatten(input_shape=(64, 64,1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5, noise_shape=None, seed=None),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5, noise_shape=None, seed=None),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.5, noise_shape=None, seed=None),
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(6,activation='softmax')])



history=model.fit(np.array(Final_array, dtype='uint8'), ok, epochs=100, validation_split=0.1,shuffle=True)

model.evaluate(np.array(Final_array, dtype='uint8'), ok, verbose=2)
