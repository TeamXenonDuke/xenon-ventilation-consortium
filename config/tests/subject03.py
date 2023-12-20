"""Test config for 007-004C, 3D radial ventilation DICOM data."""
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
        self.subject_id = "test"
        self.data_dir = "assets/tests/subject03/"
        self.xenon_dicom_dir = os.path.join(self.data_dir, "gas_recon_dicoms")
        self.proton_dicom_dir = os.path.join(self.data_dir, "proton_recon_dicoms")
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.registration_key = constants.RegistrationKey.SKIP.value
        self.scan_type = constants.ScanType.RADIAL.value
        self.processes = Process()
        self.lock()


class Process(base_config.Process):
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
