"""Segmentation module."""

import os
import pdb

import nibabel as nib
import numpy as np
import tensorflow as tf
from absl import app, flags
from scipy.ndimage import zoom
from models.serialized_batch_vnet_Relu_batch_2DGRE import vnet as vnet25D
from models.model_vnet import vnet
from utils import constants, io_utils, misc

FLAGS = flags.FLAGS
flags.DEFINE_string("image_type", "vent", "either ute or vent for segmentation")
flags.DEFINE_string("scan_type", "gre", "ether gre, spiral, or radial")
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


def predict_2d(image: np.ndarray, erosion: int = 0) -> np.ndarray:
    """Predict mask using segmentation model for 2D images.

    Args:
        image (np.ndarray): image array.
        erosion (int): kernel size for eroding mask boundary

    Returns:
        np.ndarray: predicted mask.
    """
    current_path = os.path.dirname(__file__)
    mymodel = os.path.join(
        current_path, "models", "weights", constants.CNNPaths.PROTON_2D
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

    # erode mask
    if erosion > 0:
        mask = misc.erode_image(mask, erosion)

    return mask

def predict_2dXe(ven, erosion: int = 3):
    # use SegNet model to make segmentation
    # mymodel = 'myModel_utegrow_128201.h5'
    current_path = os.path.dirname(__file__)
    mymodel = os.path.join(
        current_path, "models", "weights", constants.CNNPaths.XE_2halfD
    )

    model=vnet25D(input_size=(128,128,14,1))
    model.load_weights(mymodel);


    ven =np.abs(ven)
    ven = 255*(ven-np.min(ven))/(np.max(ven)-np.min(ven))
    

    ven = np.rot90(ven,k=-1)
    print("##################")
    print(ven.shape)
    save_real_slice = (ven.shape)[2]
    if(save_real_slice>14):
        diff_from_14 = save_real_slice-14;
        cut_at_the_end = diff_from_14//2;
        cut_at_the_start = diff_from_14-cut_at_the_end ;
        ven = ven[:,:,cut_at_the_start:save_real_slice-cut_at_the_end]
       
        
    ven_mean = np.mean(ven);
    ven_std = np.std(ven);
    print("##################")
    print("mean "+str(ven_mean)+"std "+ str(ven_std))
    print("using train mean std")
    print("##################")
    ven = (ven - ven_mean)/(ven_std)
    #ven = (ven - 8.845115957031128)/(14.890703147049313)
    ven = ven[None, ...]
    ven = ven[..., None]

    # Model Prediction
    pred_mask = model.predict(ven)
    pred_mask = pred_mask[0,...,0]


    pred_mask[pred_mask>0.5]=1;
    pred_mask[pred_mask<=0.5]=0;

    pred_mask = pred_mask.astype('float64')
    pred_mask = np.rot90(pred_mask,k=1)



    # Comment out Erosion
    #kernel = create_kernel();


    #mask_new = np.zeros([128,128,14]);


    #for ii in range(14):
    #    mask_temp = pred_mask [:,:,ii].copy();
    #    mask_temp = ndimage.binary_erosion(mask_temp, structure=kernel).astype(np.float32)
    #    mask_new[:,:,ii] = mask_temp
    #pred_mask = mask_new

    pred_mask[pred_mask>0.5]=1;
    pred_mask[pred_mask<=0.5]=0;
    ven = np.squeeze(ven)


    if(save_real_slice>14):
        mask_new = np.zeros([128,128,save_real_slice]);
        
        for ii in range(14):
            mask_new[:,:,cut_at_the_start+ii] = pred_mask [:,:,ii].copy();

        pred_mask = mask_new


    # erode mask
    if erosion > 0:
        pred_mask = misc.erode_image(pred_mask, erosion)

    
    return pred_mask


def predict_3d(
    image: np.ndarray,
    image_type: str = constants.ImageType.VENT.value,
    erosion: int = 0,
) -> np.ndarray:
    """Generate a segmentation mask from a 3D proton or ventilation image.

    Args:
        image: np.nd array of the input image to be segmented.
        image_type: str of the image type ute or vent.
    Returns:
        mask: np.ndarray of type bool of the output mask.
    """
    # get shape of the image
    img_h, img_w, _ = np.shape(image)
    # reshaping image for segmentation
    if img_h == 64 and img_w == 64:
        print("Reshaping image for segmentation")
        image = zoom(abs(image), [2, 2, 2])
    elif img_h == 128 and img_w == 128:
        pass
    else:
        raise ValueError("Segmentation Image size should be 128 x 128 x n")

    if image_type == constants.ImageType.VENT.value:
        model = vnet(input_size=(128, 128, 128, 1))
        weights_dir_current = os.path.join(
            "models", "weights", constants.CNNPaths.VENT_3D
        )
    elif image_type == constants.ImageType.UTE.value:
        model = vnet(input_size=(128, 128, 128, 1))
        weights_dir_current = os.path.join(
            "models", "weights", constants.CNNPaths.PROTON_3D
        )
    else:
        raise ValueError("image_type must be ute or vent")

    # Load model weights
    model.load_weights(weights_dir_current)

    if image_type == constants.ImageType.VENT.value:
        image = misc.standardize_image(image)
    else:
        raise ValueError("Image type must be ute or vent")
    # Model Prediction
    image = image[None, ...]
    image = image[..., None]
    mask = model.predict(image)
    # Making mask binary
    mask = mask[0, :, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask < 1] = 0
    # erode mask
    if erosion > 0:
        mask = misc.erode_image(mask, erosion)
    return mask.astype("float64")


def main(argv):
    """Run CNN model inference on ute or vent image."""
    image = nib.load(FLAGS.nii_filename).get_fdata()
    image_type = FLAGS.image_type
    scan_type = FLAGS.scan_type
    if (
        scan_type == constants.ScanType.GRE.value
        or scan_type == constants.ScanType.SPIRAL.value
    ):
        mask = predict_2d(image)
    elif scan_type == constants.ScanType.RADIAL.value:
        mask = predict_3d(image, image_type)
    export_path = os.path.dirname(FLAGS.nii_filename) + "/mask.nii"
    io_utils.export_nii(image=mask.astype("float64"), path=export_path)


if __name__ == "__main__":
    app.run(main)
