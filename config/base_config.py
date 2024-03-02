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
        self.segmentation_key = constants.SegmentationKey.CNN_VENT.value
        self.site = constants.Site.DUKE.value
        self.reference_data_key = constants.ReferenceDataKey.DEFAULT.value
        self.reference_data = ReferenceData(self.reference_data_key)
        self.subject_id = "test"
        self.xenon_dicom_dir = ""


class Process(object):
    """Define the evaluation processes."""

    def __init__(self):
        """Initialize the process parameters."""
        self.ventilation_mapping_gre = True
        self.ventilation_mapping_radial = False


class ReferenceData(object):
    """Define reference data."""

    def __init__(self, reference_data_key):
        """Initialize reference data attributes."""
        if (
            reference_data_key == constants.ReferenceDataKey.DEFAULT.value
            or reference_data_key == constants.ReferenceDataKey.MANUAL.value
        ):
            self.ref_stats_ven_dict = {
                "r_ven_defect_ave": "2.6",
                "r_ven_defect_std": "1.8",
                "r_ven_low_ave": "17.5",
                "r_ven_low_std": "5.7",
                "r_ven_high_ave": "16.7",
                "r_ven_high_std": "3.3",
                "r_ven_skewness_ave": "0",
                "r_ven_skewness_std": "0.11",
                "r_ven_CV_ave": "0.37",
                "r_ven_CV_std": "0.04",
                "r_ven_tcv_ave": "3.8",
                "r_ven_tcv_std": "0.6",
            }

            self.ref_mean_vent = 0.58
            self.ref_std_vent = 0.19
            self.ref_bins_ven = [
                self.ref_mean_vent - 2 * self.ref_std_vent,
                self.ref_mean_vent - self.ref_std_vent,
                self.ref_mean_vent,
                self.ref_mean_vent + self.ref_std_vent,
                self.ref_mean_vent + 2 * self.ref_std_vent,
            ]


def get_config() -> config_dict.ConfigDict:
    """Return the config dict. This is a required function.

    Returns:
        a ml_collections.config_dict.ConfigDict
    """
    return Config()
