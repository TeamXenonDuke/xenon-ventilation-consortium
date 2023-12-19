"""Miscellaneous util functions mostly image processing."""

import pdb
import sys

import cv2

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


def standardize_image_axes(
    image: np.ndarray, pixelsize: np.ndarray, fov: float
) -> np.ndarray:
    """Resize image such that x and y dimensions are of the same scale and length.

    Args:
        image (np.array): np.ndarray image to be scaled and resized
        pixelsize (np.array): array denoting row and column spacing of pixels in mm
        fov (float): float field of view size
    Returns:
        np.ndarray of the scaled and resized image
    """
    # resize image so that x and y array elements are spaced by 1 mm
    image_rescaled = np.zeros(
        (
            int(np.shape(image)[0] * pixelsize[1]),
            int(np.shape(image)[1] * pixelsize[0]),
            np.shape(image)[2],
        )
    ).astype("float64")
    dim = [
        int(np.shape(image)[0] * pixelsize[1]) * 1j,
        int(np.shape(image)[1] * pixelsize[0]) * 1j,
        np.shape(image)[2] * 1j,
    ]
    image_rescaled = _interpTo(image, dim)

    # if x and y dimensions are unequal, use zero padding
    image_square = np.zeros((int(fov), int(fov), np.shape(image)[2])).astype("float64")
    if np.shape(image_rescaled)[0] != int(fov) or np.shape(image_rescaled)[1] != int(
        fov
    ):
        # x or y != fov, zero fill it and make x=y
        dim_diff_x = ((np.abs(np.shape(image_rescaled)[0] - int(fov))) / 2).astype(
            "int"
        )
        dim_diff_y = ((np.abs(np.shape(image_rescaled)[1] - int(fov))) / 2).astype(
            "int"
        )

        dim_max_x = (int(fov) - dim_diff_x).astype("int")
        dim_max_y = (int(fov) - dim_diff_y).astype("int")

        image_square[dim_diff_x:dim_max_x, dim_diff_y:dim_max_y] = image_rescaled
    else:
        image_square = image_rescaled

    # resize image to standard 128 x 128 pixels
    image_standard = np.zeros((128, 128, np.shape(image)[2])).astype("float64")
    image_standard = _interpTo(
        image_square, [128j, 128j, np.shape(image)[2] * 1j]
    ).astype("float64")

    return image_standard


def get_biggest_island_indices(arr: np.ndarray) -> Tuple[int, int]:
    """Get the start and stop indices of the biggest island in the array.

    Args:
        arr (np.ndarray): binary array of 0s and 1s.
    Returns:
        Tuple of start and stop indices of the biggest island.
    """
    # intitialize count
    cur_count = 0
    cur_start = 0

    max_count = 0
    pre_state = 0

    index_start = 0
    index_end = 0
    for i in range(0, np.size(arr)):
        if arr[i] == 0:
            cur_count = 0
            if (pre_state == 1) & (cur_start == index_start):
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


def get_plot_indices(image: np.ndarray, scan_type: str) -> Tuple[int, int]:
    """Get the indices to plot the image.

    Args:
        image (np.ndarray): binary image.
        scan_type (str): scan_type
    Returns:
        Tuple of start and interval indices.
    """
    sum_line = np.sum(np.sum(image, axis=0), axis=0)
    if (
        scan_type == constants.ScanType.GRE.value
        or scan_type == constants.ScanType.SPIRAL.value
    ):
        index_start, index_end = get_biggest_island_indices(sum_line >= 0)
        flt_inter = (index_end - index_start) // constants.NUM_SLICE_GRE_MONTAGE
    elif scan_type == constants.ScanType.RADIAL.value:
        index_start, index_end = get_biggest_island_indices(sum_line > 0)
        flt_inter = (index_end - index_start) // constants.NUM_SLICE_GRE_MONTAGE

    # threshold to decide interval number
    if np.modf(flt_inter)[0] > 0.4:
        index_skip = np.ceil(flt_inter).astype(int)
    else:
        index_skip = np.floor(flt_inter).astype(int)

    # insure that index_skip is at least 1
    index_skip = max(1, index_skip)

    return index_start, index_skip


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
        binary_arr = sum_line > -1
        ind_start, ind_end = get_biggest_island_indices(binary_arr)
        flt_inter = (ind_end - ind_start) / constants.NUM_SLICE_GRE_MONTAGE

        # use 0.4 as a threshold to decide interval number
        if np.modf(flt_inter)[0] > 0.4:
            ind_inter = np.ceil(flt_inter).astype(int)
        else:
            ind_inter = np.floor(flt_inter).astype(int)
        # insure that ind_inter is at least 1
        ind_inter = max(1, ind_inter)
    else:
        raise ValueError("Invalid scan type.")
    return ind_start, ind_inter


def normalize(
    image: np.ndarray,
    mask: np.ndarray = np.array([0.0]),
    method: str = constants.NormalizationMethods.PERCENTILE_MASKED,
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
    if method == constants.NormalizationMethods.MAX:
        return image * 1.0 / np.max(image)
    elif method == constants.NormalizationMethods.PERCENTILE:
        return image * 1.0 / np.percentile(image, percentile)
    elif method == constants.NormalizationMethods.PERCENTILE_MASKED:
        image_thre = np.percentile(image[mask > 0], percentile)
        image_n = np.divide(np.multiply(image, mask), image_thre)
        image_n[image_n > 1] = 1
        return image_n
    elif method == constants.NormalizationMethods.MEAN:
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0
        return image / np.mean(image[mask > 0])
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


def erode_image(image: np.ndarray, erosion: int) -> np.ndarray:
    """Erode image.

    Erodes image slice by slice.

    Args:
        image (np.ndarray): 3-D image to erode.
        erosion (int): size of erosion kernel.
    Returns:
        Eroded image.
    """
    kernel = np.ones((erosion, erosion), np.uint8)
    for i in range(image.shape[2]):
        image[:, :, i] = cv2.erode(image[:, :, i], kernel, iterations=1)
    return image


def standardize_image(image: np.ndarray) -> np.ndarray:
    """Standardize image.

    Args:
        image (np.ndarray): image to standardize.
    Returns:
        Standardized image.
    """
    image = np.abs(image)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image - np.mean(image)) / np.std(image)
    return image
