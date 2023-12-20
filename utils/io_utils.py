"""Import and export helper functions."""

import os
import pdb
import sys

sys.path.append("..")
import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import nibabel as nib
import numpy as np
import pdf2image
import pdfkit
import pydicom
import skimage
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc, xlim, ylim

from utils import constants


def _adj_format1(x: float) -> str:
    """Convert float fraction to percentage string.

    Args:
        x (float): float value

    Returns:
        str: rounded string
    """
    num = np.around((x) * 100, decimals=0).astype(int)
    if num == 0:
        return "<1"
    else:
        return num.astype(str)


def _adj_format2(x: float) -> str:
    """Convert float to string with decimal cutoff of 2.

    Args:
        x (float): float value

    Returns:
        str: converted string
    """
    return np.around(x, decimals=2).astype(str)


def make_montage(image: np.ndarray, n_slices: int = 14) -> np.ndarray:
    """Make montage of the image.

    Makes 2xn_slices//2 montage of the image.
    Assumes the image is of shape (x, y, z, 3).
    If image contains <n_slices, blank slices are inserted

    Args:
        image (np.ndarray): image to make montage of.
        n_slices (int, optional): number of slices to plot. Defaults to 14.
    Returns:
        Montaged image array.
    """
    # get the shape of the image
    x, y, z, _ = image.shape
    # get the number of rows and columns
    n_rows = 2
    n_cols = n_slices // n_rows
    # get the shape of the slices
    slice_shape = (x, y)
    # make the montage array
    montage = np.zeros((n_rows * slice_shape[0], n_cols * slice_shape[1], 3))
    # iterate over the slices
    for i in range(n_slices):
        # get the row and column
        row = i // n_cols
        col = i % n_cols
        # get the slice
        if i < z:
            slice = image[:, :, i, :]
        else:
            slice = np.zeros((x, y, 3))
        # add to the montage
        montage[
            row * slice_shape[0] : (row + 1) * slice_shape[0],
            col * slice_shape[1] : (col + 1) * slice_shape[1],
            :,
        ] = slice

    return montage


def _merge_rgb_and_gray(gray_slice: np.ndarray, rgb_slice: np.ndarray) -> np.ndarray:
    """Combine the gray scale image with the RGB binning via HSV.

    Args:
        gray_slice (np.ndarray): 2D image slice of grayscale image.
        rgb_slice (_type_): 3D image slice of the RGB grayscale image of shape
            (H, W, C)

    Returns:
        (np.ndarray): merged image slice
    """
    # construct RGB version of gray-level ute
    gray_slice_color = np.dstack((gray_slice, gray_slice, gray_slice))
    # Convert the input image and color mask to HSV
    gray_slice_hsv = skimage.color.rgb2hsv(gray_slice_color)
    rgb_slice_hsv = skimage.color.rgb2hsv(rgb_slice)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    gray_slice_hsv[..., 0] = rgb_slice_hsv[..., 0]
    gray_slice_hsv[..., 1] = rgb_slice_hsv[..., 1]
    mask = (
        (rgb_slice[:, :, 0] == 0)
        & (rgb_slice[:, :, 1] == 0)
        & (rgb_slice[:, :, 2] == 0)
    )
    mask = ~mask
    gray_slice_hsv[mask, :] = rgb_slice_hsv[mask, :]
    colormap = skimage.color.hsv2rgb(gray_slice_hsv)
    return colormap


def importDICOM(path: str, scan_type: str) -> Dict[str, Any]:
    """Read in dicom files into a np.ndarray image.

    Args:
        path: str dicom folder path
        scan_type: int enum of the scan type (2D GRE, 2D Spiral, etc.)
    Returns:
        dict of the image and metadata
    """
    files = [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if not fname.startswith(".")
    ]

    assert len(files) > 0, "No dicom files found in the directory."
    # sort files by file name
    files = sorted(files, reverse=False)
    RefDs = pydicom.dcmread(files[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Columns), int(RefDs.Rows), len(files))

    if scan_type == constants.ScanType.GRE.value:
        slicethickness = RefDs.SpacingBetweenSlices
        pixelsize = np.array(
            RefDs.PixelSpacing
        )  # save the pixel size (#4 for vent, #2 for proton)
    elif (
        scan_type == constants.ScanType.SPIRAL.value
        or scan_type == constants.ScanType.RADIAL.value
    ):
        slicethickness = RefDs.SliceThickness
        pixelsize = np.array(
            RefDs.PixelSpacing
        )  # save the pixel size (#4 for vent, #2 for proton)

    acquisition_date = RefDs.StudyDate
    dicom = np.zeros(ConstPixelDims)
    slice_number = np.zeros(ConstPixelDims[2])
    fov = max(int(RefDs.Rows * pixelsize[0]), int(RefDs.Columns * pixelsize[1])) / 10

    for filename in files:
        # read the file
        ds = pydicom.read_file(filename)
        # get the slice number
        slice_number[files.index(filename)] = int(ds.InstanceNumber)
        # store the raw image data
        if np.std(ds.pixel_array) > 1:
            dicom[:, :, files.index(filename)] = np.transpose(ds.pixel_array, (1, 0))
    slice_order = np.argsort(slice_number)
    dicom_sorted = dicom[:, :, slice_order]

    out_dict = {
        constants.IOFields.IMAGE: dicom_sorted.astype("float64"),
        constants.IOFields.PIXEL_SIZE: pixelsize,
        constants.IOFields.SLICE_THICKNESS: slicethickness,
        constants.IOFields.FOV: fov,
        constants.IOFields.SCAN_DATE: acquisition_date,
    }
    return out_dict


def export_csv(subject_id: str, data_dir: str, stats_dict: Dict[str, Any]):
    """Export CSV file."""
    csvfile = os.path.join(
        data_dir, constants.OutputPaths.REPORT_CLINICAL + subject_id + ".csv"
    )
    with open(csvfile, "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        filewriter.writerow(
            [
                constants.STATSIOFields.SUBJECT_ID,
                constants.STATSIOFields.VEN_DEFECT,
                constants.STATSIOFields.VEN_LOW,
                constants.STATSIOFields.VEN_HIGH,
                constants.STATSIOFields.VEN_CV,
                constants.STATSIOFields.VEN_SKEW,
                constants.STATSIOFields.VEN_SNR,
                constants.STATSIOFields.INFLATION,
            ]
        )
        filewriter.writerow(
            [
                subject_id,
                stats_dict[constants.STATSIOFields.VEN_DEFECT],
                stats_dict[constants.STATSIOFields.VEN_LOW],
                stats_dict[constants.STATSIOFields.VEN_HIGH],
                stats_dict[constants.STATSIOFields.VEN_CV],
                stats_dict[constants.STATSIOFields.VEN_SKEW],
                stats_dict[constants.STATSIOFields.VEN_SNR],
                stats_dict[constants.STATSIOFields.INFLATION],
            ]
        )


def export_nii(image: np.ndarray, path: str, fov: Optional[float] = None):
    """Export image as nifti file.

    Writes voxel dimensions to nifti header if available.
    Args:
        image: np.ndarray 3D image to be exporetd
        path: str file path of nifti file
        fov: float field of view
    """
    nii_imge = nib.Nifti1Image(image, np.eye(4))

    if fov:
        # note: if non cube shape, need to verify
        # that this is the correct order of dimensions
        x = fov / np.shape(image)[0] / 10
        y = x
        z = x
        voxel_dims = [x, y, z]
        nii_imge.header["pixdim"][1:4] = voxel_dims  # type: ignore
        nib.Nifti1Image(image, np.eye(4), nii_imge.header)
    nib.save(nii_imge, path)


def export_3DRGB2nii(
    image: np.ndarray, path: str, n_slice: int, fov: Optional[float] = None
):
    """Export 4D volume to a RGB nii.

    Args:
        image (np.ndarray): 4D image of shape (128, 128, N, 3)
        path (str): file path of nifti file
        n_slice (int): number of slices to export
        fov (float): field of view. defaults to None
    """
    # need uint8 to save to RGB
    color = (np.copy(image) * 255).astype("uint8")
    color_matrix = np.copy(image).astype("uint8")
    color_slice = np.zeros((128, 128, 3)).astype("uint8")
    # some fancy and tricky re-arrange
    for i in range(0, n_slice):
        color_slice = np.copy(color[:, :, i, :])
        color_slice = np.transpose(color_slice, [2, 0, 1])
        cline = np.reshape(color_slice, (1, np.size(color_slice)))
        color_slice = np.reshape(cline, [128, 128, 3], order="A")
        color_slice = np.transpose(color_slice, [1, 0, 2])
        color_matrix[:, :, i, :] = color_slice.copy()
    # stack the RGB channels
    shape_3d = image.shape[0:3]
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    nii_data = color_matrix.copy().view(dtype=rgb_dtype).reshape(shape_3d)
    export_nii(nii_data, path, fov)


def export_montage_gray(
    image: np.ndarray,
    path: str,
    ind_start: int,
    ind_inter: int,
    rotate_img: Optional[bool] = True,
):
    """Export the grayscale image of anisotropic 2D image.

    Args:
        image: np.ndarray 3D grayscale image
        path: str output file path
        ind_start: int start index
        ind_inter: int index spacing
        rotate_img (bool): rotate image by 270 deg and flip
    """
    # divide by the maximum value
    image = image / np.max(image)

    # rotate images
    if rotate_img:
        image = np.rot90(image, 3)
        image = np.flip(image, 1)

    # stack the image to make it 4D (x, y, z, 3)
    image = np.stack((image, image, image), axis=-1)

    # plot the montage
    ind_end = ind_start + ind_inter * 14
    img_montage = make_montage(image[:, :, ind_start:ind_end:ind_inter, :])

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(img_montage, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0, dpi=300)
    plt.clf()
    plt.close()


def export_montage_overlay(
    image_bin: np.ndarray,
    image_background: np.ndarray,
    path: str,
    index2color: Dict[int, List[float]],
    ind_start: int,
    ind_inter: int,
    subject_id: str,
    slices: int = 14,
    fov: Optional[float] = None,
    rotate_img: Optional[bool] = True,
):
    """Export a montage image of the binned colormap and background anatomical image.

    Args:
        image_bin (np.ndarray): binned 3D image
        image_background (np.ndarray): background 3D image
        path (str): export file path.
        index2color (Dict[int, List[float]]): dict which maps bins to RGB color values.
        ind_start (int): starting index
        ind_inter (int): index spacing
        subject_id (str): subject ID
        slices (int, optional): Number of slices
        FOV (Optional[float], optional): Field of view. Defaults to None.
        rotate_img (bool): rotate image by 270 deg and flip

    """
    if rotate_img:
        image_bin = np.rot90(image_bin, 3)
        image_bin = np.flip(image_bin, 1)
        image_background = np.rot90(image_background, 3)
        image_background = np.flip(image_background, 1)
    # rescale background image to be between [0 ,1]
    image_background = image_background / np.percentile(image_background, 99)
    image_background[image_background > 1] = 1
    # get the image shape
    img_w, img_h, img_n = np.shape(image_bin)
    n_slice = min(img_n, constants.NUM_SLICE_GRE_MONTAGE)
    ind_end = ind_start + ind_inter * slices
    # initialize 4D image
    colormap = np.zeros((img_w, img_h, img_n, 3))

    # convert each slice from index to RGB, then combine
    # image_bin_rgb to HSV
    for slice in range(img_n):
        # convert image_bin to image_bin_rgb
        image_bin_rgb = [index2color[x] for x in image_bin[:, :, slice].flatten()]
        image_bin_rgb = np.asarray(image_bin_rgb)
        image_bin_rgb = np.reshape(image_bin_rgb, (img_w, img_h, 3))
        # merge bin_rgb with ute_reg through hsv colorspace
        colorslice = _merge_rgb_and_gray(image_background[:, :, slice], image_bin_rgb)
        colormap[:, :, slice, :] = colorslice
    nii_filename = "ven_Sub" + subject_id + ".nii"
    nii_path = os.path.join(os.path.dirname(path), nii_filename)
    export_3DRGB2nii(image=colormap, path=nii_path, fov=fov, n_slice=n_slice)
    # make montage from the image stack
    img_montage = make_montage(colormap[:, :, ind_start:ind_end:ind_inter, :])
    # plot and save the montage
    plt.figure()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(img_montage, interpolation="none")
    plt.axis("off")
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.0, dpi=300)
    plt.clf()
    plt.close()


def export_histogram(
    data: np.ndarray,
    path: str,
    color: Tuple[float, float, float],
    x_lim: float,
    y_lim: float,
    num_bins: int,
    refer_fit: Tuple[float, float, float],
    xticks=None,
    yticks=None,
):
    """Export historam plot of data along with reference histogram shape.

    Args:
        data (np.ndarray): histogram data
        path (str): export file path
        color (Tuple[float]): Tuple of the histogram color (RGB)
        x_lim (float): x limits
        y_lim (float): y limits
        num_bins (int): number of histogram bins
        refer_fit (Tuple[float]): Tuple of the reference histogram refer_fit[0] is the
            amplitude, refer_fit[1] is the shift, refer_fit[2] is the width
        xticks (_type_, optional): xtick location. Defaults to None.
        yticks (_type_, optional): ytick location. Defaults to None.
    """
    # plot histogram for the gas exchange ratio maps
    # make a thick frame

    rc("axes", linewidth=4)

    fig, ax = plt.subplots(figsize=(9, 6))
    # the histogram of the data
    # limit the range of data
    data = data.flatten()
    data[data < 0] = 0
    data[data > x_lim] = x_lim
    data = np.append(data, x_lim)
    weights = np.ones_like(data) / float(len(data))
    # plot histogram
    _, bins, _ = ax.hist(
        data, num_bins, color=color, weights=weights, edgecolor="black"
    )
    # define and plot healthy reference line
    normal = refer_fit[0] * np.exp(
        -(((np.asarray(bins) - refer_fit[1]) / refer_fit[2]) ** 2)
    )
    ax.plot(bins, normal, "--", color="k", linewidth=4)
    ax.set_ylabel("Fraction of Total Pixels", fontsize=35)
    xlim((0, x_lim))
    ylim((0, y_lim))
    plt.locator_params(axis="x", nbins=4)
    if xticks and yticks:
        xticklabels = ["{:.1f}".format(x) for x in xticks]
        yticklabels = ["{:.2f}".format(x) for x in yticks]
        plt.xticks(xticks, xticklabels, fontsize=40)
        plt.yticks(yticks, yticklabels, fontsize=40)
    else:
        logging.info("xticks or yticks not specified.")
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def export_html_pdf_vent(
    subject_id: str,
    data_dir: str,
    stats_dict: Dict[str, Any],
    ref_dict: Dict[str, Any],
    scan_type: str,
):
    """Render HTML and PDF file using templates and reference values.

    Args:
        subject_id (str): subject id name
        data_dir (str): output data directoy
        stats_dict (Dict[str, Any]): dictionary of subject stats
        ref_dict (Dict[str, Ant]): dictionary of reference stats
        scan_type (str): scan type
    """
    wd = os.path.join(os.path.dirname(__file__), "..")
    temp_clinical = os.path.join(
        wd, constants.OutputPaths.HTML_TMP, constants.OutputPaths.TEMP_GRE_CLINICAL_HTML
    )
    report_clinical = os.path.join(wd, constants.OutputPaths.REPORT_CLINICAL_HTML)
    html_dict = {
        constants.IOFields.SUBJECT_ID: subject_id,
        constants.IOFields.INFLATION: _adj_format2(
            stats_dict[constants.STATSIOFields.INFLATION]
        ),
        constants.IOFields.VEN_DEFECT: _adj_format1(
            stats_dict[constants.STATSIOFields.VEN_DEFECT]
        ),
        constants.IOFields.VEN_LOW: _adj_format1(
            stats_dict[constants.STATSIOFields.VEN_LOW]
        ),
        constants.IOFields.VEN_HIGH: _adj_format1(
            stats_dict[constants.STATSIOFields.VEN_HIGH]
        ),
        constants.IOFields.VEN_MEAN: _adj_format2(
            stats_dict[constants.STATSIOFields.VEN_MEAN]
        ),
        constants.IOFields.VEN_CV: _adj_format2(
            stats_dict[constants.STATSIOFields.VEN_CV]
        ),
        constants.IOFields.VEN_SKEW: _adj_format2(
            stats_dict[constants.STATSIOFields.VEN_SKEW]
        ),
        constants.IOFields.VEN_SNR: _adj_format2(
            stats_dict[constants.STATSIOFields.VEN_SNR]
        ),
        constants.IOFields.VEN_COR_MONTAGE: os.path.join(
            data_dir, constants.OutputPaths.VEN_COR_MONTAGE_PNG
        ),
        constants.IOFields.RAW_PROTON_MONTAGE: os.path.join(
            data_dir, constants.OutputPaths.PROTON_REG_MONTAGE_PNG
        ),
        constants.IOFields.VEN_MONTAGE: os.path.join(
            data_dir, constants.OutputPaths.VEN_COLOR_MONTAGE_PNG
        ),
        constants.IOFields.VEN_HIST: os.path.join(
            data_dir, constants.OutputPaths.VEN_HIST_PNG
        ),
        constants.IOFields.SCAN_DATE: stats_dict[constants.STATSIOFields.SCAN_DATE],
        constants.IOFields.PROCESS_DATE: stats_dict[
            constants.STATSIOFields.PROCESS_DATE
        ],
        constants.IOFields.SCAN_TYPE: scan_type.upper(),
    }

    html_dict.update(ref_dict)
    with open(temp_clinical, "r") as f:
        data = f.read()
        rendered = data.format(**html_dict)
    with open(report_clinical, "w") as o:
        o.write(rendered)
    # generate pdf from html
    pdf_clinical_path = os.path.join(
        data_dir, constants.OutputPaths.REPORT_CLINICAL + "_" + subject_id + ".pdf"
    )
    logging.info("exporting report into pdf file")
    options = constants.PDFOPTIONS.VEN_PDF_OPTIONS
    pdfkit.from_file(report_clinical, pdf_clinical_path, options=options)
    pages = pdf2image.convert_from_path(pdf_clinical_path, 500)
    for page in pages:
        jpg_path = os.path.join(
            data_dir, constants.OutputPaths.REPORT_CLINICAL + "_" + subject_id + ".jpg"
        )
        page.save(jpg_path, "JPEG")
    os.remove(report_clinical)
