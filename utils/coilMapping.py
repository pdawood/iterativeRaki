#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:39:53 2022

@author: ubuntu
"""
import numpy as np
from sigpy import mri as mr

def getMask(acs): 
    '''
    Function to generate mask. Calculation based on ESPiRIT. 
    Args:
        acs: Auto-Calibration-Signal in shape [coil, PE, RO].
    Returns:
        mask.
    '''
    coilMaps = mr.app.EspiritCalib(acs)
    coilMaps = coilMaps.run()
   
    mask_ = np.asarray(np.where(coilMaps[10, :, :] != 0))
    mask = np.zeros(coilMaps[10, :, :].shape)
    mask[mask_[0], mask_[1]] = 1
    
    return mask
