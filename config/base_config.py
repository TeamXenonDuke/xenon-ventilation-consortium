"""Base configuration file."""
import sys

from ml_collections import config_dict

# parent directory
sys.path.append("..")

from utils import constants


class Config(config_dict.ConfigDict):
    """Base config file.

    Attributes:
        bias_key: the bias field correction method to be used.
        data_dir: the directory containing the data to be exported.
        manual_reg_dir: the file path to the manual registration file.
        manual_seg_dir: the file path to the manual segmentation file.
        processes: the processes to be evaluated.
        proton_dicom_dir: the file path to the proton dicom directory.
        registration_key: the registration method to be used.
        scan_type: the scan type.
        segmentation_key: the segmentation method to be used.
        site: the site.
        subject_id: the subject id.
        xenon_dicom_dir: the file path to the xenon dicom directory.
    """

    def __init__(self):
        """Initialize config parameters."""
        super().__init__()
        self.bias_key = constants.BiasfieldKey.N4ITK.value
        self.data_dir = ""
        self.manual_reg_dir = ""
        self.manual_seg_dir = ""
        self.processes = Process()
        self.proton_dicom_dir = ""
        self.registration_key = constants.RegistrationKey.PROTON2GAS.value
        self.scan_type = constants.ScanType.GRE.value
        self.segmentation_key = constants.SegmentationKey.CNN_PROTON.value
        self.site = constants.Site.DUKE.value
        self.subject_id = "test"
        self.xenon_dicom_dir = ""


class Process(object):
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
