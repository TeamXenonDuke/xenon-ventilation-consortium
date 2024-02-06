"""Launch GUI to run the pipeline."""

import logging
import os
import pdb
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk

from absl import app

from config import base_config
from main_mapping import ventilation_mapping_gre
from utils import constants


class GUI(tk.Tk):
    """Guided user interface for ventilation mapping."""

    def __init__(self):
        """Initialize the GUI."""
        super().__init__()

        # window settings
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = sw / 2, sh / 2
        x, y = (sw - w) / 2, (sh - h) / 2
        self.geometry("%dx%d+%d+%d" % (w, h, x, y))
        self.configure(width=sw / 2, height=sh / 2)
        self.title("Xenon Ventilation Mapping GUI")

        # configure column weights
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

        # create the config class
        self.config = base_config.Config()

        # create the StringVars
        self.var_biasfield_flag = tk.StringVar()
        self.var_dicomdir_proton = tk.StringVar()
        self.var_dicomdir_xe = tk.StringVar()
        self.var_initialdir = tk.StringVar()
        self.var_manual_registration_file = tk.StringVar()
        self.var_manual_segmentation_file = tk.StringVar()
        self.var_outputdir = tk.StringVar()
        self.var_registration_flag = tk.StringVar()
        self.var_scan_type = tk.StringVar()
        self.var_segmentation_flag = tk.StringVar()
        self.var_site = tk.StringVar()
        self.var_subject_id = tk.StringVar()

        # create the widgets
        self.create_widgets()

    def create_widgets(self):
        """Create all the widgets."""
        padding = {"padx": 2, "pady": 5}
        # labels
        ttk.Label(self, text="Set Subject ID").grid(row=0, column=0, **padding)
        ttk.Label(self, text="Set Scan Type").grid(row=1, column=0, **padding)
        ttk.Label(self, text="Set Site").grid(row=2, column=0, **padding)
        ttk.Label(self, text="Set Segmentation flag").grid(row=3, column=0, **padding)
        ttk.Label(self, text="Set Registration flag").grid(row=4, column=0, **padding)
        ttk.Label(self, text="Set Biasfield Correction flag").grid(
            row=5, column=0, **padding
        )
        ttk.Label(self, text="Set ventilation directory").grid(
            row=6, column=0, **padding
        )
        ttk.Label(self, text="Set proton directory").grid(row=7, column=0, **padding)
        ttk.Label(self, text="Set output directory").grid(row=8, column=0, **padding)

        # entries
        ttk.Entry(self, textvariable=self.var_subject_id).grid(
            row=0, column=1, **padding
        )
        tk.Entry(
            self,
            textvariable=self.var_dicomdir_xe,
            width=50,
        ).grid(row=6, column=1, **padding)
        tk.Entry(self, textvariable=self.var_dicomdir_proton, width=50).grid(
            row=7, column=1, **padding
        )
        tk.Entry(self, textvariable=self.var_outputdir, width=50).grid(
            row=8, column=1, **padding
        )
        # dropdown menus
        ttk.OptionMenu(
            self,
            self.var_scan_type,
            constants.ScanType.GRE.value,
            *[_.value for _ in constants.ScanType]
        ).grid(row=1, column=1, **padding)

        ttk.OptionMenu(
            self,
            self.var_site,
            constants.Site.DUKE.value,
            *[_.value for _ in constants.Site]
        ).grid(row=2, column=1, **padding)

        ttk.OptionMenu(
            self,
            self.var_segmentation_flag,
            constants.SegmentationKey.CNN_VENT.value,
            *[_.value for _ in constants.SegmentationKey]
        ).grid(row=3, column=1, **padding)

        ttk.OptionMenu(
            self,
            self.var_registration_flag,
            constants.RegistrationKey.PROTON2GAS.value,
            *[_.value for _ in constants.RegistrationKey]
        ).grid(row=4, column=1, **padding)

        ttk.OptionMenu(
            self,
            self.var_biasfield_flag,
            constants.BiasfieldKey.N4ITK.value,
            *[_.value for _ in constants.BiasfieldKey]
        ).grid(row=5, column=1, **padding)

        # buttons
        ttk.Button(
            self, text="Select .nii file", command=self.select_segmentation_file
        ).grid(row=3, column=2, **padding)
        ttk.Button(
            self, text="Select .nii file", command=self.select_registration_file
        ).grid(row=4, column=2, **padding)
        ttk.Button(self, text="Select folder", command=self.select_xe_directory).grid(
            row=6, column=2, **padding
        )
        ttk.Button(
            self, text="Select folder", command=self.select_proton_directory
        ).grid(row=7, column=2, **padding)
        ttk.Button(
            self, text="Select folder", command=self.select_output_directory
        ).grid(row=8, column=2, **padding)
        ttk.Button(self, text="Quit", command=self.quit).grid(
            row=10, column=0, **padding
        )
        tk.Button(self, text="Execute", command=self.execute, fg="blue").grid(
            row=10, column=1, **padding
        )

    def select_segmentation_file(self):
        """Open file dialog to set path to manual segmentation mask file."""
        filename = fd.askopenfilename(
            filetypes=[(".nii files", "*.nii")], initialdir=self.var_initialdir.get()
        )
        self.var_manual_segmentation_file.set(filename)
        self.var_initialdir.set(os.path.dirname(filename))

    def select_registration_file(self):
        """Open file dialog to set path to manual segmentation mask file."""
        filename = fd.askopenfilename(
            filetypes=[(".nii files", "*.nii")], initialdir=self.var_initialdir.get()
        )
        self.var_manual_registration_file.set(filename)
        self.var_initialdir.set(os.path.dirname(filename))

    def select_xe_directory(self):
        """Open file dialog to set xenon images dicom directory."""
        directory = fd.askdirectory(initialdir=self.var_initialdir.get())
        self.var_dicomdir_xe.set(directory)
        self.var_initialdir.set(os.path.dirname(directory))

    def select_proton_directory(self):
        """Open file dialog to set proton images dicom directory."""
        directory = fd.askdirectory(initialdir=self.var_initialdir.get())
        self.var_dicomdir_proton.set(directory)
        self.var_initialdir.set(os.path.dirname(directory))

    def select_output_directory(self):
        """Open file dialog to set file output directory."""
        directory = fd.askdirectory(initialdir=self.var_initialdir.get())
        self.var_outputdir.set(directory)
        self.var_initialdir.set(os.path.dirname(directory))

    def prepare_config(self):
        """Store the user-selected parameters into the config file."""
        self.config.bias_key = self.var_biasfield_flag.get()
        self.config.data_dir = self.var_outputdir.get()
        self.config.manual_reg_dir = self.var_manual_registration_file.get()
        self.config.manual_seg_dir = self.var_manual_segmentation_file.get()
        self.config.proton_dicom_dir = self.var_dicomdir_proton.get()
        self.config.registration_key = self.var_registration_flag.get()
        self.config.scan_type = self.var_scan_type.get()
        self.config.segmentation_key = self.var_segmentation_flag.get()
        self.config.site = self.var_site.get()
        self.config.subject_id = self.var_subject_id.get()
        self.config.xenon_dicom_dir = self.var_dicomdir_xe.get()

    def update_initialdir(self):
        """Update the initial dialog directory to be the output directory."""
        self.var_initialdir.set(self.var_outputdir.get())

    def execute(self):
        """Run the pipleline."""
        self.prepare_config()
        self.update_initialdir()
        if self.config.processes.ventilation_mapping_gre:
            logging.info("2D Ventilation mapping.")
            ventilation_mapping_gre(self.config)


def main(argv):
    """Create the GUI elements."""
    gui = GUI()
    gui.mainloop()


if __name__ == "__main__":
    app.run(main)
