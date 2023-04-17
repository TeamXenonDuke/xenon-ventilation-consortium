"""Script to run the 2D ventilation pipeline."""
import logging
import pdb

from absl import app, flags
from ml_collections import config_dict, config_flags

from classmap_gre import GRESubject

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config", None, "config file.")


def ventilation_mapping_gre(config: config_dict.ConfigDict):
    """Run the 2D ventilation pipeline.

    Args:
        config (config_dict.ConfigDict): config dict
    """
    subject = GRESubject(config=config)
    # read in dicom files
    logging.info("1. reading in files")
    subject.readinfiles()
    logging.info("2. Registering and Segmenting Proton Mask")
    subject.segmentation()
    subject.registration()
    logging.info("3. Bias Field Correction")
    subject.biasfield_correction()
    logging.info("4. Binning and Ventilation Mask")
    subject.gas_binning()
    subject.generate_statistics()
    logging.info("5. Generate Clinical Report")
    subject.generate_figures()
    subject.generateHtmlPdf()
    logging.info("6. Exporting .mat and .csv files")
    subject.generateCSV()
    subject.savefiles()
    logging.info("Complete")


def main(argv):
    """Run the 2D ventilation pipeline."""
    config = _CONFIG.value
    if config.processes.ventilation_mapping_gre:
        logging.info("2D Ventilation mapping.")
        ventilation_mapping_gre(config)


if __name__ == "__main__":
    app.run(main)
