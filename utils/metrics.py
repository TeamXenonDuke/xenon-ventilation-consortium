"""Metrics utility functions."""
import math
import sys

sys.path.append("..")
import pdb
from typing import Tuple

import numpy as np
import scipy
import scipy.stats
from scipy import ndimage

from utils import constants


def fSNR_3T(image: np.ndarray, mask: np.ndarray) -> Tuple[float, ...]:
    """Calculate SNR and Rayleigh SNR.

    Args:
        image: np.ndarray image to calculate SNR
        mask: np.ndarray mask to calculate SNR inside

    Returns: Tuple of the SNR, Rayleigh SNR, image signal, image noise
    """
    # x-y border exclusion
    xybe = 0
    mini_cube_dim = [8, 8, 8]
    my_dim = np.shape(image)

    # dilate the mask to analyze noise area away from the signal
    def util(x):
        return int((math.ceil(x * 0.025) * 2 + 1))

    dilate_struct = np.ones((util(my_dim[0]), util(my_dim[1]), util(my_dim[2])))
    noise_mask = ndimage.morphology.binary_dilation(mask, dilate_struct).astype(bool)
    # exclude border too
    if xybe > 0:
        noise_mask[0:xybe, :, :] = True
        noise_mask[-xybe:, :, :] = True
        noise_mask[:, 0:xybe, :] = True
        noise_mask[:, -xybe:, :] = True

    noise_temp = np.copy(image)
    noise_temp[noise_mask] = np.nan
    # set up for using mini noise cubes to go through the image and calculate std for noise
    (mini_x, mini_y, mini_z) = mini_cube_dim

    n_noise_vox = mini_x * mini_y * mini_z

    mini_vox_std = 0.75 * n_noise_vox  # minimul number of voxels to calculate std

    stepper = 0
    total = 0
    std_dev_mini_noise_vol = []

    for ii in range(0, int(my_dim[0] / mini_x)):
        for jj in range(0, int(my_dim[1] / mini_y)):
            for kk in range(0, int(my_dim[2] / mini_z)):
                mini_cube_noise_dist = noise_temp[
                    ii * mini_x : (ii + 1) * mini_x,
                    jj * mini_y : (jj + 1) * mini_y,
                    kk * mini_z : (kk + 1) * mini_z,
                ]

                mini_cube_noise_dist = mini_cube_noise_dist[
                    ~np.isnan(mini_cube_noise_dist)
                ]

                # only calculate std for the noise when it is long enough
                if len(mini_cube_noise_dist) > mini_vox_std:
                    std_dev_mini_noise_vol.append(np.std(mini_cube_noise_dist, ddof=1))
                    stepper = stepper + 1

                total = total + 1

    image_noise = float(np.median(std_dev_mini_noise_vol))
    image_signal = float(np.average(image[mask]))
    SNR = float(image_signal / image_noise) if image_noise > 0 else np.inf
    SNR_Rayleigh = float(SNR * constants.RAYLEIGH_FACTOR)
    return SNR, SNR_Rayleigh, image_signal, image_noise


def binStats(image, image_bin, image_raw, mask):
    """Generate statistics from the binned image.

    Args:
        image: np.ndarray 3D image volume
        image_bin: np.ndarray 3D binned image
        image_raw: np.ndarray reconstructed image data, no post processing
        mask: np.ndarray boolean thoracic cavity mask
    """
    stats_dict = {}
    maskall = np.sum(mask).astype("float")

    stats_dict[constants.STATSIOFields.VEN_DEFECT] = np.divide(
        np.sum((image_bin == 2)), maskall
    )
    stats_dict[constants.STATSIOFields.VEN_LOW] = np.divide(
        np.sum((image_bin == 3)), maskall
    )
    stats_dict[constants.STATSIOFields.VEN_MEAN] = np.average(image[mask])
    stats_dict[constants.STATSIOFields.VEN_MEDIAN] = np.median(image[mask])
    stats_dict[constants.STATSIOFields.VEN_STD] = np.std(image[mask])
    stats_dict[constants.STATSIOFields.VEN_CV] = (np.std(abs(image[mask]))) / (
        np.average(abs(image[mask]))
    )
    stats_dict[constants.STATSIOFields.VEN_HIGH] = np.divide(
        np.sum((image_bin == 6) | (image_bin == 7)), maskall
    )

    _, SNR_Rayleigh, _, _ = fSNR_3T(image_raw, mask)
    stats_dict[constants.STATSIOFields.VEN_SNR] = SNR_Rayleigh
    stats_dict[constants.STATSIOFields.VEN_SKEW] = scipy.stats.skew(image[mask])
    return stats_dict


def inflation_volume_2D(mask: np.ndarray, fov: float, slice_thickness: float) -> float:
    """Calculate the inflation volume of anisotropic image.

    Args:
        mask: np.ndarray thoracic cavity mask
        fov: float field of view
        slice_thickness: float slice thickness
    """
    return np.multiply(
        np.sum(mask),
        (fov**2 / np.shape(mask)[0] ** 2 / constants.FOVINFLATIONSCALE2D)
        * slice_thickness,
    )
