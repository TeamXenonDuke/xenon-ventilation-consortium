"""Define important constants used throughout the pipeline."""
import enum

FOVDIMSCALE = 10
FOVINFLATIONSCALE2D = 10000.0
FOVINFLATIONSCALE3D = 1000.0

NUM_SLICE_GRE_MONTAGE = 14
NUM_ROWS_GRE_MONTAGE = 2
NUM_COLS_GRE_MONTAGE = 7

DEFAULT_SLICE_THICKNESS = 3.125
DEFAULT_PIXEL_SIZE = 3.125
DEFAULT_MAX_IMG_VALUE = 255.0

RAYLEIGH_FACTOR = 0.66
VEN_PERCENTILE_RESCALE = 99.0
VEN_PERCENTILE_THRESHOLD_SEG = 80
PROTON_PERCENTILE_RESCALE = 99.8


class IOFields(object):
    """General IOFields constants."""

    BIASFIELD_KEY = "biasfield_key"
    FOV = "fov"
    IMAGE = "image"
    INFLATION = "inflation"
    MASK_REG_NII = "mask_reg_nii"
    OUTPUT_PATH = "output_path"
    PIXEL_SIZE = "pixel_size"
    PROCESS_DATE = "process_date"
    PROTON_DICOM_DIR = "proton_dicom_dir"
    PROTON_REG_NII = "proton_reg_nii"
    RAW_PROTON_MONTAGE = "raw_proton_montage"
    REFERENCE_DATA_KEY = "reference_data_key"
    REGISTRATION_KEY = "registration_key"
    SCAN_DATE = "scan_date"
    SCAN_TYPE = "scan_type"
    SEGMENTATION_KEY = "segmentation_key"
    SITE = "site"
    SLICE_THICKNESS = "slice_thickness"
    SUBJECT_ID = "subject_id"
    VEN_COR_MONTAGE = "bias_cor_ven_montage"
    VEN_CV = "ven_cv"
    VEN_DEFECT = "ven_defect"
    VEN_HIGH = "ven_high"
    VEN_HIST = "ven_hist"
    VEN_LOW = "ven_low"
    VEN_MEAN = "ven_mean"
    VEN_MEDIAN = "ven_median"
    VEN_MONTAGE = "ven_montage"
    VEN_SKEW = "ven_skewness"
    VEN_SNR = "ven_snr"
    VEN_STD = "ven_std"
    VENT_DICOM_DIR = "vent_dicom_dir"


class OutputPaths(object):
    """Output file names."""

    GRE_MASK_NII = "GRE_mask.nii"
    GRE_REG_PROTON_NII = "GRE_regproton.nii"
    GRE_VENT_BINNING_NII = "GRE_ventbinning.nii"
    GRE_VENT_COR_NII = "GRE_ventcor.nii"
    GRE_VENT_RAW_NII = "GRE_ventraw.nii"
    HTML_TMP = "html_tmp"
    PROTON_REG_MONTAGE_PNG = "raw_proton_montage.png"
    REPORT_CLINICAL = "report_clinical"
    REPORT_CLINICAL_HTML = "report_clinical.html"
    TEMP_GRE_CLINICAL_HTML = "temp_clinical_gre.html"
    VEN_COLOR_MONTAGE_PNG = "ven_montage.png"
    VEN_COR_MONTAGE_PNG = "bias_cor_ven_montage.png"
    VEN_HIST_PNG = "ven_hist.png"
    VEN_RAW_MONTAGE_PNG = "raw_ven_montage.png"


class CNNPaths(object):
    """Paths to saved model files."""

    PROTON_2D = "GREModel_20190323.h5"
    PROTON_3D = "model_ANATOMY_UTE.h5"
    VENT_3D = "model_ANATOMY_VEN.h5"
    XE_2_5_D = "model_ANATOMY_VEN_2DGRE.h5"

class ImageType(enum.Enum):
    """Segmentation flags."""

    VENT = "vent"
    UTE = "ute"


class SegmentationKey(enum.Enum):
    """Segmentation flags.

    Defines how and if thoracic cavity segmentation is performed. Options:
    - CNN_VENT: Deep learning segmentation on xenon ventilation image.
    - CNN_PROTON: Deep learning segmentation on proton image.
    - MANUAL_VENT: Read in Nifti file of mask manually segmented on xenon image.
    - MANUAL_PROTON: Read in Nifti file of mask manually segmented on proton image.
    - THRESHOLD_VENT: Perform thresholding on xenon ventilation image. Typically used
        for debugging.
    """

    CNN_VENT = "cnn_vent"
    CNN_PROTON = "cnn_proton"
    MANUAL_VENT = "manual_vent"
    MANUAL_PROTON = "manual_proton"
    THRESHOLD_VENT = "threshold_vent"


class ReferenceDataKey(enum.Enum):
    """Reference data flags.

    Defines which reference data to use. Options:
    GRE_2D: Reference data for 2D GRE
    RADIAL_2D: Rerence data for 3D radial
    DEFAULT: Default reference data used
    MANUAL: Use when manualy adjusting default reference data
    """

    MANUAL = "manual"
    DEFAULT = "default"


class RegistrationKey(enum.Enum):
    """Registration flags.

    Defines how and if registration is performed. Options:
    PROTON2GAS: Register ANTs to register proton image (moving) to gas image (fixed).
        Also uses the transformation and applies on the mask if segmented on proton
        image.
    MASK2GAS: Register ANTs to register mask (moving) to gas image (fixed).
        Also uses the transformation and applies on the proton image.
    MANUAL: Read in Nifti file of manually registered proton image.
    SKIP: Skip registration entirely.
    """

    MANUAL = "manual"
    MASK2GAS = "mask2gas"
    PROTON2GAS = "proton2gas"
    SKIP = "skip"


class BiasfieldKey(enum.Enum):
    """Biasfield correction flags.

    Defines how and if biasfield correction is performed. Options:
    N4ITK: Use N4ITK bias field correction.
    SKIP: Skip bias field ocrrection entirely.
    """

    N4ITK = "n4itk"
    SKIP = "skip"


class ScanType(enum.Enum):
    """Scan type."""

    GRE = "gre"
    SPIRAL = "spiral"
    RADIAL = "radial"


class Site(enum.Enum):
    """Scan type."""

    DUKE = "duke"
    UVA = "uva"


class STATSIOFields(object):
    """Statistic IO Fields."""

    INFLATION = "inflation"
    PROCESS_DATE = "process_date"
    SCAN_DATE = "scan_date"
    SUBJECT_ID = "subject_id"
    VEN_CV = "ven_cv"
    VEN_DEFECT = "ven_defect"
    VEN_HIGH = "ven_high"
    VEN_LOW = "ven_low"
    VEN_MEAN = "ven_mean"
    VEN_MEDIAN = "ven_median"
    VEN_SKEW = "ven_skewness"
    VEN_SNR = "ven_snr"
    VEN_STD = "ven_std"


class VENHISTOGRAMFields(object):
    """Ventilation historam fields."""

    COLOR = (0.4196, 0.6824, 0.8392)
    XLIM = 1.0
    YLIM = 0.07
    NUMBINS = 50
    REFERENCE_FIT = (0.04462, 0.52, 0.2713)


class PDFOPTIONS(object):
    """PDF Options dict."""

    VEN_PDF_OPTIONS = {
        "page-width": 256,  # 320,
        "page-height": 160,  # 160,
        "margin-top": 1,
        "margin-right": 0.1,
        "margin-bottom": 0.1,
        "margin-left": 0.1,
        "dpi": 300,
        "encoding": "UTF-8",
        "enable-local-file-access": None,
    }


class NormalizationMethods(object):
    """Image normalization methods."""

    MAX = "max"
    PERCENTILE_MASKED = "percentile_masked"
    PERCENTILE = "percentile"
    MEAN = "mean"


class BIN2COLORMAP(object):
    """Maps of binned values to color values."""

    VENT_BIN2COLOR_MAP = {
        1: [0, 0, 0],
        2: [1, 0, 0],
        3: [1, 0.7143, 0],
        4: [0.4, 0.7, 0.4],
        5: [0, 1, 0],
        6: [0, 0.57, 0.71],
        7: [0, 0, 1],
    }
