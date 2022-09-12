import numpy as np

def getPseudoKernel(num_hid_layer, layer_design):
    '''
    Function to calculate effective kernel-size in multi-layer Convolution-Networks.
    
    Args:
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture.

    Returns:
        p_pseudo, r_pseudo: effective convolution kernel dimension in PE - and RO-
        direction, respectively.
    '''    
    p_pseudo = layer_design[1][1][0]
    r_pseudo = layer_design[1][1][1]

    if num_hid_layer > 1:
        for i in range(2, num_hid_layer + 1):
            p_pseudo += layer_design[i][1][0] - 1
            r_pseudo += layer_design[i][1][1] - 1

    p_pseudo += layer_design['output_unit'][1][0] - 1
    r_pseudo += layer_design['output_unit'][1][1] - 1

    return p_pseudo, r_pseudo


def extractDatCNN(acs, R, num_hid_layer, layer_design):
    '''
    Function to get Training-Data for CNNs for k-space interpolation. 
    Args: 
        acs: Auto-Calibration-Signal in shape [coil, PE, RO].
        R: Undersampling Factor.
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture.

    Returns:
        prc_data: Dictionary with keys 'src': Source-Signals.
                                        'trg': Target-Signals.
    '''    
    Nk_p, Nk_r = getPseudoKernel(num_hid_layer, layer_design)

    # number of coils, phase-steps, read-out-steps
    (num_coils, num_p_acs, num_r_acs) = acs.shape

    # computing how many times the kernel fits into the block
    # in read-direction (rep_r_acs) and in phase-direction (rep_p_acs)
    # for acs-block
    rep_r_acs = (num_r_acs - Nk_r) + 1
    rep_p_acs = num_p_acs - ((Nk_p - 1) * (R - 1) + Nk_p) + 1

    kernel_extension_p = (Nk_p - 1) * (R - 1) + Nk_p

    # determining the k-points within the kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    p_trg = Nk_p // 2 - 1

    # dimension of target-vector
    dim_trg_vec = (R - 1) * acs.shape[0]

    # initializing the source - and target - matrices obtained from acs-block
    trg_data = np.ndarray((rep_p_acs, rep_r_acs, dim_trg_vec),
                          dtype=complex)

    # loop for kernel-displacement in phase-direction
    for p in range(rep_p_acs):
        # loop for kernel-displacement in read-direction
        for r in range(rep_r_acs):
            trg_data[p, r, ] = acs[:, p+p_trg*R+1:p+p_trg*R+R:1, r+r_trg
                                  ].reshape((dim_trg_vec))

    trg_data = np.expand_dims(trg_data, axis=1)

    src_data = np.ndarray((rep_p_acs,
                           num_coils,
                           kernel_extension_p,
                           num_r_acs),
                           dtype=complex
                           )

    for i in range(rep_p_acs):
        src_data[i, :, :, :] = acs[:, i:i+kernel_extension_p:1, :]
    
    src_data = src_data.transpose((0, 2, 3, 1))
 
    prc_data = {'src' : src_data, 'trg' : trg_data}
    
    return prc_data


def fillDatCNN(kspace_zf, pred_mat, R, num_hid_layer, layer_design):
    '''
    Function to re-insert estimated missing signals back into zero-filled k-space.
    Args:
        kspace_zf: Zerofilled kspace.
        pred_mat: Estimated signals.
        R: Undersampling Factor.
        num_hid_layer: Number of 'hidden' layers in Network. 
        layer_design: Dictionary describing Network-Architecture.
    
    Returns: 
        Reconstructed k-space in shape [coils, PE, RO].
    '''
    Nk_p, Nk_r = getPseudoKernel(num_hid_layer, layer_design)
    # number of coils, phase-steps, read-out-steps
    (num_coil, num_p_data, num_r_data) = kspace_zf.shape

    # for data-block
    rep_p = (num_p_data - ((Nk_p - 1) * (R - 1) + Nk_p))//R + 1
    rep_r = num_r_data - Nk_r + 1

    # determining the k-points within the kernel which are related to
    # the target points
    r_trg = Nk_r//2
    p_trg = Nk_p//2 - 1

    for p in range(rep_p):
        for r in range(rep_r):
            kspace_zf[:, p*R+p_trg*R+1:p*R+p_trg*R +
                 R:1, r+r_trg] = pred_mat[R*p, r, ].reshape((num_coil, R-1, ))

    return kspace_zf
