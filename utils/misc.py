"""Miscellaneous util functions mostly image processing."""

import pdb
import sys

sys.path.append("..")
from typing import Any, List, Tuple

import numpy as np
import scipy
import skimage
from scipy import ndimage

from utils import constants


def _interp3(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Sample 3D array v using linear interpolation.

    Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to scipy.ndimage.map_coordinates.

    Args:
        x: np.ndarray x corner locations
        y: np.ndarray y corner locations
        z: np.ndarray z corner locations
        v: np.ndarray 3D image volume to be interpolated
        xi: np.ndarray x grid points to be sampled
        yi: np.ndarray y grid points to be sampled
        zi: np.ndarray z grid points to be sampled
    """

    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    ndimage.map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def _interpTo(image: np.ndarray, dim: List[Any]):
    """Interpolate image to specified dimensions.

    Args:
        image: np.ndarray image to be interpolated
        dim: List[int] output image dimensions
    """
    # Interpolate the GRE proton to 128^3
    x = np.linspace(1, 128, np.shape(image)[0])
    y = np.linspace(1, 128, np.shape(image)[1])
    z = np.linspace(
        1, 128, np.shape(image)[2]
    )  # spreading the 14 slices across 128 slices

    # desired output shape 128*128*128, the "128j" here means 128 steps
    # (from 1 to 128, take 128 steps)
    xi, yi, zi = np.mgrid[1 : 128 : dim[0], 1 : 128 : dim[1], 1 : 128 : dim[2]]
    interpolated = _interp3(x, y, z, image, xi, yi, zi)
    return interpolated


def scale2match(image: np.ndarray, pixelsize: float, fov: float) -> np.ndarray:
    """Scale and resize ventilation image to numpy array.
    TODO: Rename this function properly.

    Args:
        image: np.ndarray image to be scaled and resized
        pixelsize: float size of pixel
        fov: float field of view size
    Returns:
        np.ndarray of the scaled and resized image
    """
    ## crop the ventilation image to x=y and scale to 128x128
    scaled_image = np.zeros((128, 128, np.shape(image)[2])).astype("float64")
    image_fullsize = np.zeros(
        (
            round(np.shape(image)[0] * pixelsize),
            round(np.shape(image)[1] * pixelsize),
            np.shape(image)[2],
        )
    ).astype("float64")
    image_final = np.zeros((int(fov), int(fov), np.shape(image)[2])).astype("float64")
    dim = [
        round(np.shape(image)[0] * pixelsize) * 1j,
        round(np.shape(image)[1] * pixelsize) * 1j,
        np.shape(image)[2] * 1j,
    ]
    image_fullsize = _interpTo(image, dim)

    if np.shape(image_fullsize)[0] != int(fov) or np.shape(image_fullsize)[1] != int(
        fov
    ):
        # x or y != fov, zero fill it and make x=y
        dim_diff_x = ((np.abs(np.shape(image_fullsize)[0] - int(fov))) / 2).astype(
            "int"
        )
        dim_diff_y = ((np.abs(np.shape(image_fullsize)[1] - int(fov))) / 2).astype(
            "int"
        )

        dim_max_x = (int(fov) - dim_diff_x).astype("int")
        dim_max_y = (int(fov) - dim_diff_y).astype("int")

        image_final[dim_diff_x:dim_max_x, dim_diff_y:dim_max_y] = image_fullsize
        # scale
        scaled_image = _interpTo(image_final, [128j, 128j, np.shape(image)[2] * 1j])
    else:
        scaled_image = _interpTo(image_fullsize, [128j, 128j, np.shape(image)[2] * 1j])

    return scaled_image.astype("float64")


def _get_index_max_ones(arr: np.ndarray) -> Tuple[int, int]:
    """Calculate starting index and ending index of the max consecutive ones.

    Args:
        arr: 1D array
    """
    cur_count = 0
    cur_start = 0

    max_count = 0
    pre_state = 0

    index_start = 0
    index_end = 0

    for i in range(0, np.size(arr)):
        if arr[i] == 0:
            cur_count = 0
            if (pre_state == 1) and (cur_start == index_start):
                index_end = i - 1
            pre_state = 0

        else:
            if pre_state == 0:
                cur_start = i
                pre_state = 1
            cur_count += 1
            if cur_count > max_count:
                max_count = cur_count
                index_start = cur_start

    return index_start, index_end


def get_start_interval(mask: np.ndarray, scan_type: str) -> Tuple[int, int]:
    """Determine the starting slice index and the interval to display the montage.

    Args:
        mask: np.ndarray thoracic cavity mask
        scan_type: str scan type
    """
    sum_line = np.sum(np.sum(mask, axis=0), axis=0)
    if (
        scan_type == constants.ScanType.GRE.value
        or scan_type == constants.ScanType.SPIRAL.value
    ):
        num_slice = constants.NUM_SLICE_GRE_MONTAGE
        binary_arr = sum_line > -1
    else:
        raise ValueError("Invalid scan type.")
    ind_start, ind_end = _get_index_max_ones(binary_arr)

    flt_inter = (ind_end - ind_start) / num_slice

    # use 0.4 as a threshold to decide interval number
    if np.modf(flt_inter)[0] > 0.4:
        ind_inter = np.ceil(flt_inter).astype(int)
    else:
        ind_inter = np.floor(flt_inter).astype(int)
    # insure that ind_inter is at least 1
    ind_inter = max(1, ind_inter)
    return ind_start, ind_inter


def normalize(
    image: np.ndarray,
    method: str,
    mask: np.ndarray = np.array([0.0]),
    percentile: float = 99.0,
) -> np.ndarray:
    """Normalize the image to be between [0, 1.0].

    Args:
        image (np.ndarray): image matrix
        method (int): normalization method
        mask (np.ndarray): boolean mask
        percentile (np.ndarray): if normalization via max percentile, scale by
            percentile and set everything else to 1.0

    Returns:
        np.ndarray: normalized image
    """
    if method == constants.NormalizationMethods.VANILLA:
        return image * 1.0 / np.max(image)
    elif method == constants.NormalizationMethods.PERCENTILE:
        image_thre = np.percentile(image[mask], percentile)
        image_n = np.divide(np.multiply(image, mask), image_thre)
        image_n[image_n > 1] = 1
        return image_n
    else:
        raise ValueError("Invalid normalization method")


def remove_small_objects(mask: np.ndarray, scale: float = 0.1):
    """Remove small unconnected voxels in the mask.

    Args:
        mask (np.ndarray): boolean mask
        scale (float, optional): scalaing factor to determin minimum size.
            Defaults to 0.015.

    Returns:
        Mask with the unconnected voxels removed
    """
    min_size = np.sum(mask) * scale
    return skimage.morphology.remove_small_objects(
        ar=mask, min_size=min_size, connectivity=1
    ).astype("bool")
