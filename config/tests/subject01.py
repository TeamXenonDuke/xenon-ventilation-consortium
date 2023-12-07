"""Test config for 13740-22."""
import os
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")
from config import base_config
from utils import constants


class Config(base_config.Config):
    """Base config file."""

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.data_dir = "assets/tests/subject01/"
        self.manual_reg_dir = ""
        self.manual_seg_dir = os.path.join(self.data_dir, "mask_manual.nii")
        self.processes = Process()
        self.proton_dicom_dir = os.path.join(
            self.data_dir, "1H_SSFSECORONALBODY_2x2x15_5"
        )
        self.registration_key = constants.RegistrationKey.PROTON2GAS.value
        self.scan_type = constants.ScanType.GRE.value
        self.segmentation_key = constants.SegmentationKey.CNN_PROTON.value
        self.site = constants.Site.DUKE.value
        self.subject_id = "13740-22"
        self.xenon_dicom_dir = os.path.join(self.data_dir, "gre_hpg_cor_2102_10")
        self.lock()


class Process(base_config.Process):
    """Define the evaluation processes."""

    def __init__(self):
        """Initialize the process parameters."""
        self.ventilation_mapping_gre = True
        self.ventilation_mapping_radial = False


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
