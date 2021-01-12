#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:06:12 2020

@author: sctuser
"""

import glob
import os
import subprocess

for dti in glob.glob('/media/josh/Josh/Ari_btfeeight/03078890btfF/merge.nii.gz'):
    dti_path, vol = os.path.split(dti)
    
    print('starting denoise on {}'.format(dti_path))
    subprocess.call(['/home/josh/sct_dev/bin/sct_maths','-i', dti,
            '-denoise', '1', '-o' ,os.path.join(dti_path,'denoise.nii')], cwd=dti_path)
    
    fbvec = glob.glob(os.path.join(dti_path, 'bvec2.bvec'))[0]
    fbval = glob.glob(os.path.join(dti_path, '*.bval'))[0]
    
    denoise = glob.glob(os.path.join(dti_path, 'denoise.nii'))[0]
    print('starting moco on {}'.format(dti_path))
    subprocess.call(['/home/josh/sct_dev/bin/sct_dmri_moco', '-i', 
            denoise, '-bvec', fbvec, '-bval', 
            fbval, '-x', 'spline'], cwd=dti_path)
    
    print('starting dti estimation on {}'.format(dti_path))
    subprocess.call(['/home/josh/sct_dev/bin/sct_dmri_compute_dti', '-i',
            os.path.join(dti_path, 'denoise_moco.nii'), '-bval', fbval, '-bvec', 
            fbvec, '-o', 'DTI'],cwd=dti_path)