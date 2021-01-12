# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from glob import glob

spath = 'C:/SC_AI2/Russell/AI_DTI_2/DTI_3'
nmaps = 2
crop_size = 16
ASIA = json.loads(open('C:/SC_AI2/labels.json').read())      
n = 0
asia_slices = []
D = np.zeros((5695,crop_size*2,crop_size*2,nmaps))
scale_factor = [1,1000]

for assec in ASIA.keys():
    mask_name = glob(os.path.join(spath,assec,'*ACTUAL_DTI*/denoise_moco_dwi_mean_seg.nii'))[0]
    mask_img = nib.load(mask_name)
    mask = mask_img.get_fdata()
    V = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2],nmaps))
    FA_name = glob(os.path.join(spath,assec,'*ACTUAL_DTI*/*ACTUAL_DTIFA.nii.gz'))[0]    
    FA_img = nib.load(FA_name)    
    V[:,:,:,0] = FA_img.get_fdata()
    AD_name = glob(os.path.join(spath,assec,'*ACTUAL_DTI*/*ACTUAL_DTIAD.nii.gz'))[0]    
    AD_img = nib.load(AD_name)    
    V[:,:,:,1] = AD_img.get_fdata()
    RD_name = glob(os.path.join(spath,assec,'*ACTUAL_DTI*/*ACTUAL_DTIRD.nii.gz'))[0]    
    RD_img = nib.load(RD_name)    
    V[:,:,:,1] = RD_img.get_fdata()
    MD_name = glob(os.path.join(spath,assec,'*ACTUAL_DTI*/*ACTUAL_DTIMD.nii.gz'))[0]    
    MD_img = nib.load(MD_name)    
    V[:,:,:,3] = MD_img.get_fdata()
    for mm in range(0,nmaps):
        V[:,:,:,mm][mask<1] = 0
    
    
    vlev = int(ASIA[assec]["L"])
    
    
    cutoff = [0,65]
        
    for zz in range(0,V.shape[2]):
        
        mask_slice = mask[:,:,zz]
        
        if np.max(mask_slice)>0:
            y,x = np.nonzero(mask_slice)
            ymean = y.mean().astype('int8')
            xmean = x.mean().astype('int8')
            
            if ymean + crop_size > mask_slice.shape[0]:
                ymean =  mask_slice.shape[0] -crop_size
            elif ymean - crop_size < 0:
                ymean = crop_size
            if xmean + crop_size > mask_slice.shape[1]:
                xmean =  mask_slice.shape[1] -crop_size
            elif xmean - crop_size < 0:
                xmean = crop_size
            for mm in range(0,nmaps):
                current_slice = V[:,:,zz,mm]
                D[n,:,:,mm] = current_slice[ymean-crop_size:ymean+crop_size,xmean-crop_size:xmean+crop_size]*scale_factor[mm]
            if zz in range(cutoff[0],cutoff[1]):
                if int(ASIA[assec]["S"])<2:
                    asia_slices.append(0)
                else:
                    asia_slices.append(1)
                #asia_slices.append(ASIA[assec]["S"])
            else:
                asia_slices.append(2)
            n = n + 1
    
            
L = np.asarray(asia_slices,dtype='int8')