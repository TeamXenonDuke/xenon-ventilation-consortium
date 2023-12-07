"""Test config for CTSA 41 from UVA E-cig study."""
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")
import os

from utils import constants


class Config(config_dict.ConfigDict):
    """Base config file."""

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.data_dir = "assets/tests/subject02/"
        self.manual_reg_dir = ""
        self.manual_seg_dir = (
            "/mnt/d/Patients/102007_w24/Dedicated_Ventilation/GRE_mask_manual.nii"
        )
        self.processes = Process()
        self.proton_dicom_dir = os.path.join(
            self.data_dir, "GRE_1H_VENT_COR_BREATH-HOLD_0004"
        )
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.scan_type = constants.ScanType.GRE.value
        self.segmentation_key = constants.SegmentationKey.CNN_PROTON.value
        self.site = constants.Site.UVA.value
        self.subject_id = "CTSA-41"
        self.xenon_dicom_dir = os.path.join(
            self.data_dir, "GRE_129XE_VENT_COR_BREATH-HOLD_0005"
        )
        self.lock()


class Process(object):
    """Define the evaluation processes."""

    def __init__(self):
        """Initialize the process parameters."""
        super().__init__()
        self.ventilation_mapping_gre = True
        self.ventilation_mapping_radial = False


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
