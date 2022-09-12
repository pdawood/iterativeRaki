import numpy as np 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def ssim(gt: np.ndarray, pred: np.ndarray, maxval = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    ssim = structural_similarity(gt, pred,
                                 data_range=gt.max())
    return ssim 

