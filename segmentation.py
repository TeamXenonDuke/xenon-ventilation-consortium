"""Segmentation module."""

import os
import pdb

import nibabel as nib
import numpy as np
import tensorflow as tf
from absl import app, flags

from utils import constants, io

FLAGS = flags.FLAGS
flags.DEFINE_string("image_type", "vent", "either ute or vent for segmentation")
flags.DEFINE_string("nii_filename", "", "nii image file path")


def hist_transform(img: np.ndarray) -> np.ndarray:
    """Transform image histogram.

    Args:
        img (np.ndarray): input image

    Returns:
        np.ndarray: transformed image
    """
    img = img / img.max()
    img = 511 * img
    img_16b = img.astype(np.uint16)
    all_pixels = img_16b.flatten()

    img_r, img_c, img_s = np.shape(img_16b)
    max_intn = np.max(img_16b)
    mu_i = 0.25 * max_intn
    mu_s = 355
    pc1 = 0.001
    pc2 = 0.998
    s1 = 0
    s2 = 511
    p1 = pc1 * max_intn
    p2 = pc2 * max_intn

    pixel_list1 = all_pixels[all_pixels <= mu_i]
    pixel_list2 = all_pixels[all_pixels > mu_i]

    standarized_list1 = np.round(
        mu_s + (pixel_list1 - mu_i) * (s1 - mu_s) / (p1 - mu_i)
    )
    standarized_list2 = np.round(
        mu_s + (pixel_list2 - mu_i) * (s2 - mu_s) / (p2 - mu_i)
    )

    standarized_values = np.zeros(np.shape(all_pixels))
    standarized_values[all_pixels <= mu_i] = standarized_list1
    standarized_values[all_pixels > mu_i] = standarized_list2
    transformed_data = np.reshape(standarized_values, [img_r, img_c, img_s])

    return transformed_data.astype("float64")


def evaluate(image: np.ndarray) -> np.ndarray:
    """Predict mask using segmentation model.

    Args:
        image (np.ndarray): image array.

    Returns:
        np.ndarray: predicted mask.
    """
    current_path = os.path.dirname(__file__)
    mymodel = os.path.join(
        current_path, "models", "weights", constants.CNNPaths.DEFAULT
    )
    image = np.rot90(image)
    ute_trans = hist_transform(image)
    n_labels = 2
    img_h, img_w, img_d = np.shape(ute_trans)
    mask = np.zeros(np.shape(ute_trans))
    if img_h != 128 or img_w != 128:
        raise Exception("Segmentation Image size should be 128 x 128 x n")

    # a deep matrix: in 3rd dimension, the label will be set to 1
    def de_label_map(myPred):
        myPred = np.reshape(myPred, [img_w, img_h, n_labels])
        return np.argmax(myPred, axis=2)  # collapse the 3rd dimension

    autoencoder = tf.keras.Sequential()
    autoencoder = tf.keras.models.load_model(mymodel)

    for i in range(0, img_d):
        ute_slice = np.flipud(ute_trans[:, :, i])
        ute_thre = np.percentile(ute_slice, 99)
        ute_slice = np.divide(ute_slice, ute_thre)
        ute_slice[ute_slice > 1] = 1
        ute_slice = np.multiply(ute_slice, 255)
        mask_slice = autoencoder.predict(np.reshape(ute_slice, (1, img_w, img_h, 1)))  # type: ignore
        mask_slice = de_label_map(mask_slice)
        mask_slice = np.rot90(np.flipud(mask_slice), 3)
        mask[:, :, i] = mask_slice
    mask = mask.astype("float64")

    return mask


def main(argv):
    """Run CNN model inference on ute or vent image."""
    image = nib.load(FLAGS.nii_filename).get_fdata()
    image_type = FLAGS.image_type
    mask = evaluate(image)
    export_path = os.path.dirname(FLAGS.nii_filename) + "/mask.nii"
    io.export_nii(image=mask.astype("float64"), path=export_path)


if __name__ == "__main__":
    app.run(main)
