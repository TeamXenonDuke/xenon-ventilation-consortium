"""Base configuration file."""
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
        self.data_dir = "/mnt/d/Patients/102009_W00/Dedicated_Ventilation"
        self.manual_reg_dir = ""
        self.manual_seg_dir = os.path.join(self.data_dir, "GRE_mask_manual.nii")
        self.processes = Process()
        self.proton_dicom_dir = os.path.join(
            self.data_dir, "gre_proton_cor_2102_consort_2"
        )
        self.registration_key = constants.RegistrationKey.PROTON2GAS.value
        self.scan_type = constants.ScanType.GRE.value
        self.segmentation_key = constants.SegmentationKey.MANUAL_VENT.value
        self.site = constants.Site.UVA.value
        self.subject_id = "102009_W00"
        self.xenon_dicom_dir = os.path.join(self.data_dir, "xe_gre_hpg_cor_2102_new_7")
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
