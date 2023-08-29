# -*- coding: utf-8 -*-
"""
@author: Christian Schrader
@email: christian.schrader@ptb.de
"""

import os
import sys
import numpy as np

from camnl import ImportNLCalibData_Base, NLInputData, NLInputAVGData, ImageRoi
from lmk_image import lmk_image
from collections import OrderedDict
import json

import logging

import PIL

logging.getLogger("PIL").setLevel(logging.WARNING)

# from libtiff import TIFF, TIFFfile, TIFFimage


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    # format='%(name)s (%(lineno)d) %(levelname)s - %(message)s',
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("camlin")
# logging.getLogger("matplotlib.font_manager").disabled = True


class ImportNLCalibData_lmk(ImportNLCalibData_Base):
    """Import data from a directory tree with TechnoTeam-LMK pus-Files.

        General Usage
        =============

        Create Object with basic settings: dirname, which groups (integration
        time series) to use, binning size:

            nlcd = nlCalibData(dirname=path_in, bright_series=groups_nl, dxdy=dxdy))

        Import dark data (with lens cap). Allowed names in self._allowed_dark_series.

            dat_dark_short = nlcd.read_dark_data('dark_short')
            dat_dark_long = nlcd.read_dark_data('dark_long')

        Import bright data.
            dat_bright = nlcd.read_bright_data()

        Calculate y_ad from dark data:
            y0 = nlcal.calc_yad(dat_dark_short)[:, 0]

        Select usable data by value (eg. cut off data with overload):

            dat_sel = nlcal.select_linearity_data(dat_bright)

        -------------------  FIXME docstring


        Class to read in nonlinearity-measurement data frpm HDF5 files.

        Parameters
        ----------
        dirname : str, required
            name of directory containing data
        bright_series : list of str, optional
            selection of named measurement groups to use.The default is None,
            whicht means all groups.
        overload_value : int, optional
            limit that defines max of A/D converter output. It is not
            distinguishable between exact this value and larger ones.
            The default is 4095.
        dxdy : 2-tuple of int, optional
            size of averaging region. The default is (1,1).
        mode : TYPE, optional
            evaluation mode. The default depends of dxdy.

        Returns
        -------
        None.

    """

    dirname: str = ""
    input_mode: str = ""
    _allowed_dark_series: tuple = ("dark", "dark_short", "dark_long")
    _allowed_input_modes: tuple = ("averaged", "samples")
    # fnprefix: str = "nonlin"

    def __init__(self, dirname, input_mode="samples", **kwargs):
        self.dirname = dirname
        assert input_mode in self._allowed_input_modes
        self.input_mode = input_mode
        super().__init__(**kwargs)
        return

    def get_image_shape(self):
        """Determine image shape of current dataset."""
        # open first image in first series
        ds_name = self.get_series_available()[0]
        if self.input_mode == "samples":
            fn_img = os.path.join(self.dirname, ds_name, "0", "0.pus")
            shape = self._read_lmk_image(fn_img)[0].shape
        elif self.input_mode == "averaged":
            fn_img = os.path.join(self.dirname, ds_name, "0_mean.tiff")
            shape = self._read_tiff_image(fn_img).shape
        else:
            raise Exception("Unknown input_mode!")
        return shape

    # add default values
    def set_bright_series(self, bright_series=None):
        if bright_series is None:
            bright_series = self.get_bright_series_available()
        self.bright_series = bright_series
        return

    def set_dark_series(self, dark_series=None):
        if dark_series is None:
            dark_series = ["dark_short"]
        self.dark_series = dark_series
        return

    # ---- Read functions

    def read_bright_data(self):
        """Read bright data."""
        if self.bright_series is None or len(self.bright_series) == 0:
            raise Exception("No bright series set!")

        # tic = time.perf_counter()
        dat = []
        for s in self.bright_series:
            d = self.read_inttime_series(s, mode=self.mode)
            dat.append(d)
        # toc = time.perf_counter()
        # logger.debug("Dauer Einlesen: %.1f" % (toc-tic) )
        return dat

    def read_dark_data(self):
        """Read dark data."""
        if self.dark_series is None or len(self.dark_series) == 0:
            raise Exception("No dark series set!")

        dat = []
        for s in self.dark_series:
            d = self.read_inttime_series(s, mode=self.mode)
            dat.append(d)
        return dat

    def read_inttime_series(self, series_name, mode=None):
        """Read all data of inttime series with repetitive measurements.

        storage structure:
        .../<series_name>/<inttime_nr>/img_<nr>

        target shape: (region, inttime, rep. measurement)

        Rearrange data. The 2D structure of the image gets flattened to 1D

        - axis 0: evaluation regions / pixel
        - axis 1: inttime steps
        - axis 2: samples, can be just one, if averaged
        """
        if mode is None:
            mode = self.mode

        assert mode in self._allowed_modes
        logger.debug("Series: %s Mode: %s" % (series_name, mode))

        series_dir = os.path.join(self.dirname, series_name)

        measurement_data = json.load(
            open(os.path.join(series_dir, "measurement.json")),
            object_pairs_hook=OrderedDict,
        )
        self.measurement_data = measurement_data
        measurement_data["nr_samples"]

        dat_samples = []
        # erstmal nur samples
        for inttime_nr, inttime_real in measurement_data["inttimes"].items():
            img_mean = img_var = img_ovrld = header = None

            samples_dir = os.path.join(series_dir, inttime_nr)
            # here different storage formats, tt-lmk-images for raw samples and
            # tiff images for averaged images
            if self.input_mode == "samples":
                img_stack = self._read_lmk_dataset(samples_dir)
            elif self.input_mode == "averaged":
                img_mean, img_var, img_ovrld, header = self._read_tiff_dataset(
                    samples_dir
                )
                img_stack = img_mean[np.newaxis, ::] # with just 1 element
            else:
                raise Exception("Unknown input_mode!")

            self.dbg_img_stack = img_stack  # only debug

            if __debug__:
                print(".", end="")

            """
            FIXME: bisher wird nur dat_samples gesammelt und nach dat.samples gespeichert
            Bei averaging muss auch var, n und ovld gespeichert werden
            Wie befüllt man die Daten nach dat, so dass man hinterher sinnvoll mit
            weiterarbeiten kann?


            """

            # distinction of evaluation mode "pixel" vs. "region" allows speed
            # optimizations for "pixel". For dxdy==(1,1) the result should be the same!
            if mode == "pixel":
                regiondat = img_stack.reshape(
                    (img_stack.shape[0], img_stack.shape[1] * img_stack.shape[2])
                ).T  # transponieren. dann sind die Pixel in erster Dimension, die Wiederholungen in zweiter
                dat_samples.append(regiondat)
                # self.regiondat = regiondat
            elif mode == "region":
                # durch das Loopen über die ROIs werden die Daten automatisch
                # geflattet
                # 2.6s !!!
                # regiondat = np.asarray([img[:, roi.begin_y:roi.end_y+1, roi.begin_x:roi.end_x+1].ravel() for roi in self._rois])

                # kaum schneller, 2.4s
                roi = self._rois[0]
                regiondat = np.ndarray(
                    (len(self._rois), img_stack.shape[0] * roi.size()),
                    dtype=img_stack.dtype,
                )
                for i, roi in enumerate(self._rois):
                    regiondat[i] = img_stack[
                        :,
                        roi.begin_y : roi.end_y + 1,
                        roi.begin_x : roi.end_x + 1,
                    ].ravel()

                dat_samples.append(regiondat)
                self.regiondat = regiondat

            elif mode == "image":  # no reordering, just access the image(s)
                dat_samples.append(img_stack)
                # FIXME: put this elsewhere? doesn't fit here

            else:
                raise Exception("Unknown averaging method: %s" % (mode))

        if __debug__:
            print()  # finish dot line

        dat = NLInputData()
        dat.series_name = series_name
        if mode == "image":
            dat.samples = np.asarray(dat_samples)
        else:
            dat.samples = np.asarray(dat_samples).swapaxes(
                0, 1
            )  # dann sind regions in erster Dimension, was ist sinnvoll?
        dat.inttimes = np.asarray(list(measurement_data["inttimes"].values()))
        # dat.rois is debug-info for line picker module, not used for nl calculations
        # store this information here is simpler than reconstructing it later
        dat.rois = [
            ImageRoi(
                begin_x=r.begin_x + self._img_roi.begin_x,
                end_x=r.end_x + self._img_roi.begin_x,
                begin_y=r.begin_y + self._img_roi.begin_y,
                end_y=r.end_y + self._img_roi.begin_y,
            )
            for r in self._rois
        ]
        return dat

    def _read_lmk_dataset(self, dirname):
        # print("OpenLmkDS: ", fn)
        img_stack = []
        for f in os.listdir(dirname):  # order is not important
            if not f.endswith(".pus"):
                continue
            img, header = self._read_lmk_image(os.path.join(dirname, f))

            # slice as early as possible
            sly = slice(self._img_roi.begin_y, self._img_roi.end_y + 1)
            slx = slice(self._img_roi.begin_x, self._img_roi.end_x + 1)
            img_stack.append(img[sly, slx])
        img_stack = np.asarray(img_stack)
        return img_stack

    def _read_tiff_dataset(self, fn):
        # print("OpenTiffDS: ", fn)
        fn_mean, fn_var, fn_ovrld, fn_hdr = self._get_tiff_filenames(fn)
        img_mean = self._read_tiff_image(fn_mean)
        img_var = self._read_tiff_image(fn_var)
        img_ovrld = self._read_tiff_image(fn_ovrld)
        sly = slice(self._img_roi.begin_y, self._img_roi.end_y + 1)
        slx = slice(self._img_roi.begin_x, self._img_roi.end_x + 1)
        img_mean = img_mean[sly, slx]  # not allowed to be None
        if img_var is not None:
            img_var = img_var[sly, slx]
        if img_ovrld is not None:
            img_ovrld = img_ovrld[sly, slx]
        header = {}
        with open(fn_hdr) as f:
            for line in f.readlines():
                k, v = line.split("=")
                header[k] = v.strip()
        return img_mean, img_var, img_ovrld, header

    def _read_lmk_image(self, fn):
        # print("  OpenLMK: ", fn)
        lmkimg = lmk_image(fn)
        return lmkimg.get(), lmkimg.get_header()

    def _read_tiff_image(self, fn):
        # print("  OpenTiff: ", fn)
        if not os.path.isfile(fn):
            return None
        return np.asarray(PIL.Image.open(fn))

    def _get_tiff_filenames(self, fn):
        """Build filenames for tiff-datasets.

        fn can be like <path>/0 or <poth>/0_[mean|var}.tiff or <path>/0_hdr.dat
        """
        if fn.endswith(("_mean.tiff", "_var.tiff", "_ovrld.tiff", "_hdr.dat")):
            # cut out everything before last underscore
            fn = fn[0 : len(fn) - fn[::-1].find("_") - 1]
        return fn + "_mean.tiff", fn + "_var.tiff", fn + "_ovrld.tiff", fn + "_hdr.dat"

    # ---- Misc functions

    def get_series_available(self, dirname=None):
        """Listet die Namen aller Messserien im Datenfile auf."""
        if dirname is None:
            dirname = self.dirname
        assert dirname is not None

        os.path.isdir(dirname)

        series = []
        for fn in os.listdir(dirname):
            fn_dir = os.path.join(dirname, fn)
            if os.path.isdir(fn_dir):
                series.append(fn)
        return series

    def get_bright_series_available(self, dirname=None):
        """Listet die Namen aller Hell-Messungen im Datenfile auf."""
        # FIXME: wozu war die Einschränkung auf self.bright_series?
        # return [k for k in self.get_series_available(dirname) if k not in self.dark_series and k in self.bright_series]
        return [
            k
            for k in self.get_series_available(dirname)
            if k not in self._allowed_dark_series
        ]

    def get_dark_series(self, dirname=None):
        """Listet die Namen aller Dunkel-Messungen im Datenfile auf."""
        return [
            k
            for k in self.get_series_available(dirname)
            if k in self._allowed_dark_series
        ]

    def get_label_from_dirname(self, fn=None):
        """Determin camera label from dataset dirname.

        Local dirname convention is '<prefix>_<camera label>.h5'.
        Default prefix is 'nonlin'.
        """
        if fn is None:
            fn = self.dirname
        label = os.path.splitext(os.path.basename(fn))[0]
        # cut off prefix if given
        prefix = ""
        if self.fnprefix is not None:
            prefix = self.fnprefix + "_"
        if label.startswith(prefix):
            label = label[len(prefix) :]
        return label
