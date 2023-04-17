"""Binning and statistics operations."""
import pdb
import sys

sys.path.append("..")
from typing import List, Tuple

import numpy as np

from utils import constants, misc


def _binning(image: np.ndarray, bin_threshold: List[float]) -> np.ndarray:
    """Bin the image.

    Args:
        image: np.ndarray image to be binned
        bin_threshold: List binning thresholds

    Returns: np.ndarray of the binned image
    """
    bvolume = np.ones(np.shape(image))
    bvolume[(image > 0) & (image <= bin_threshold[0])] = 2
    for k in range(len(bin_threshold) - 1):
        bvolume[(image > bin_threshold[k]) & (image <= bin_threshold[k + 1])] = k + 3
    bvolume[image > bin_threshold[-1]] = len(bin_threshold) + 2
    return bvolume


def gasBinning(
    image: np.ndarray,
    bin_threshold: List[float],
    mask: np.ndarray,
    percentile: float = 99,
) -> Tuple[np.ndarray, ...]:
    """Rescale and bin the image given the bin_threshold.

    Args:
        image: np.ndarray image to be binned
        bin_threshold: List bin_threshold
        mask: np.ndararay masked region to performed binning
    Returns:
        Tuple of the rescaled image, binned image, and mask that excludes VDP
    """
    # rescale
    image_n = misc.normalize(
        image,
        method=constants.NormalizationMethods.PERCENTILE,
        mask=mask.astype(bool),
        percentile=percentile,
    )
    gas_binning = _binning(image_n, bin_threshold)
    # create ventilation mask
    mask_vent = np.copy(gas_binning)
    # exclude VDP in the ventilation map
    mask_vent[mask_vent < 3] = 0
    mask_vent = mask_vent.astype(bool)
    return image_n, gas_binning, mask_vent
