#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:43:46 2022
@author: ubuntu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from .utils.complexCNN import complexNet
from .utils.shapeAnalysis import extractDatCNN, fillDatCNN

RAKI_RECO_DEFAULT_LR = 0.0005
RAKI_RECO_DEFAULT_EPOCHS = 500

IRAKI_RECO_DEFAULT_INIT_LR = 5e-4
IRAKI_RECO_DEFAULT_LR_DECAY = {
    4 : 3e-5,
    5 : 4e-5,
    6 : 6e-5
}
IRAKI_RECO_DEFAULT_ACS_NUM = 65


def rakiReco(kspc_zf, acs, R, layer_design):
    '''
    This function trains RAKI, and puts the interpolated signals 
    into zero-filled k-space.
    
    Args:
        kspc_zf: Zero-filled k-space, not including acs, in shape [coils, PE, RO].
        acs: Auto-Calibration-Signal, in shape [coils, PE, RO].
        R: Undersampling-Factor.
        layer_design: Network-Architecture. Here is a example with two hidden layers:

        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }
    
    Returns:
        network_reco: RAKI k-space reconstruction, in shape [coils, PE, RO].
    '''
    print('Starting Standard RAKI...')
    # Get Source- and Target Signals
    prc_data = extractDatCNN(acs,
                             R=R,
                             num_hid_layer=layer_design['num_hid_layer'],
                             layer_design=layer_design)
    
    trg_kspc = prc_data['trg'].transpose((0, 3, 1, 2))
    src_kspc = prc_data['src'].transpose((0, 3, 1, 2))

    src_kspc = torch.from_numpy(src_kspc).type(torch.complex64)
    trg_kspc = torch.from_numpy(trg_kspc).type(torch.complex64)
    
    net = complexNet(layer_design, R=R)    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=RAKI_RECO_DEFAULT_LR) 
      
    for _ in trange(RAKI_RECO_DEFAULT_EPOCHS):    
        optimizer.zero_grad()
        pred_kspc = net(src_kspc)['tot']
        loss =   (criterion(pred_kspc.real, trg_kspc.real)
                + criterion(pred_kspc.imag, trg_kspc.imag)) 
        loss.backward()
        optimizer.step()

    kspc_zf_input = np.expand_dims(kspc_zf, axis=0)    
    kspc_zf_input = torch.from_numpy(kspc_zf_input).type(torch.complex64)    
    # Estimate missing signals 
    kspc_pred = net(kspc_zf_input)['tot']
    kspc_pred = kspc_pred.detach().numpy()
    kspc_pred = kspc_pred.transpose((0, 2, 3, 1))
    kspc_pred = np.squeeze(kspc_pred)

    # Put estimated signals bach into zero-filled kspace 
    network_reco = fillDatCNN(kspc_zf,
                              kspc_pred,
                              R,
                              num_hid_layer=layer_design['num_hid_layer'],
                              layer_design=layer_design)
    print('Finished Standard RAKI...')
    return network_reco


def irakiReco(kspc_zf, acs, R, layer_design, grappa_reco, acs_flag, acs_start, acs_end):
    '''
    This function trains iterative-RAKI, and puts the interpolated signals 
    into zero-filled k-space.
    
    Args:
        kspc_zf: Zero-filled k-space, not including acs, in shape [coils, PE, RO].
        acs: Auto-Calibration-Signal, in shape [coils, PE, RO]
        R: Undersampling-Factor.
        layer_design: Network-Architecture. Here is a example with two hidden layers:
        
        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }
        
        grappa_reco: GRAPPA k-space reconstruction, in shape [coil, PE, RO].
        acs_flag: Set true, when acs re-inserted into reconstructed k-space.
        acs_start: Index. 
        acs_end: Index.
    
    Returns:
        network_reco: Iterative RAKI k-space reconstruction, in shape [coils, PE, RO].
    '''    
    print('Starting Iterative RAKI...')
    # Define amount of central lines as extended ACS region
    num_Acs_big = IRAKI_RECO_DEFAULT_ACS_NUM
    # Define initial learning rate at first iteration step
    lrg_init = IRAKI_RECO_DEFAULT_INIT_LR
    # Get decay of learning rate after each iteration step
    try:
        d_lrg = IRAKI_RECO_DEFAULT_LR_DECAY[R]
    except KeyError:
        return

    # Get total number of iterations
    Niter = int(lrg_init // d_lrg)

    (nC, nP, nR) = kspc_zf.shape

    # Get extended acs from central grappa lines    
    acs_data_big = grappa_reco[:, int(nP / 2 - num_Acs_big / 2):int(nP / 2 + num_Acs_big / 2), :]

    kspc_zf_input = np.expand_dims(kspc_zf, axis=0)    
    kspc_zf_input = torch.from_numpy(kspc_zf_input).type(torch.complex64)
    
    for i in range(Niter):
        print('Iteration ', str(i + 1), ' of ', str(Niter))
        # Get Target - and Source Signals     
        prc_data = extractDatCNN(acs_data_big,
                                 R=R,
                                 num_hid_layer=layer_design['num_hid_layer'],
                                 layer_design=layer_design)
        trg_kspc = prc_data['trg'].transpose((0, 3, 1, 2))
        src_kspc = prc_data['src'].transpose((0, 3, 1, 2))

        if i == 0:
            net = complexNet(layer_design, R=R)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lrg_init) # 0.0005
    
        # training
        src_kspc = torch.from_numpy(src_kspc).type(torch.complex64)
        trg_kspc = torch.from_numpy(trg_kspc).type(torch.complex64)

        # optional for time benefit: variable epoch count depending on iteration number
        EPOCHS = RAKI_RECO_DEFAULT_EPOCHS if i == 0 else 250
        
        for _ in trange(EPOCHS):    
            optimizer.zero_grad()
            pred_kspc = net(src_kspc)['tot']
            loss =   (criterion(pred_kspc.real, trg_kspc.real)
                    + criterion(pred_kspc.imag, trg_kspc.imag)) 
            loss.backward()
            optimizer.step()
            
        kspc_pred = net(kspc_zf_input)['tot']
        kspc_pred = kspc_pred.detach().numpy()
        kspc_pred = kspc_pred.transpose((0, 2, 3, 1))
        kspc_pred = np.squeeze(kspc_pred)

        networkReco_iraki = fillDatCNN(
            kspc_zf, kspc_pred, R,
            num_hid_layer=layer_design['num_hid_layer'],
            layer_design=layer_design
        )
    
        if acs_flag:
            networkReco_iraki[:, acs_start:acs_end+1, :] = acs
        
        acs_data_big = networkReco_iraki[:, int((nP - num_Acs_big) / 2):int((nP + num_Acs_big) / 2), :]
                           
        print('Current Learning rate:', '{:.5f}'.format(optimizer.param_groups[0]['lr']))
        optimizer.param_groups[0]['lr'] -= d_lrg
        print('Updated Learning rate:', '{:5f}'.format(optimizer.param_groups[0]['lr']))      

    print('Finished Iterative RAKI...')
    return networkReco_iraki
