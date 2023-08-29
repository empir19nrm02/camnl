# -*- coding: utf-8 -*-
"""
@author: Christian Schrader
@email: christian.schrader@ptb.de
"""

# import os
import sys
from collections import namedtuple
import numpy as np

# import scipy as sp
# import time
# import numpy.polynomial.polynomial as po

# import numpy.polynomial as pol
# from scipy.interpolate import CubicSpline  # , interp1d

# import scipy.stats
# import warnings

# from dataclasses import dataclass

# from functools import partial
# from scipy import integrate
# from scipy.optimize import least_squares

# from scipy.sparse import lil_matrix, coo_matrix

import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    # format='%(name)s (%(lineno)d) %(levelname)s - %(message)s',
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("camnl")
# logging.getLogger("matplotlib.font_manager").disabled = True


# attributes are immutable!
ImageRoi = namedtuple(
    "ImageRoi",
    ["begin_x", "end_x", "begin_y", "end_y"],
    defaults=[None, None, None, None],
)
ImageRoi.size = lambda roi: (roi.end_x - roi.begin_x + 1) * (
    roi.end_y - roi.begin_y + 1
)

# ------------------------- Import Classes ----------------------------------


class ImportNLCalibData_Base:
    """Base class for loading and storing input data for NL-calibration.

    This class defines the API and some helper functions. It is meant to be
    derived by child classes that implement the access to data.

    Parameters
    ----------
    avgreg_size : 2-tuple
        Defines the size of averaging regions. This can be used to aggregate
        some pixels to a 'macro pixel' to save memory or reduce noise.
        if avgreg_size is (1,1) and mode is False, then mode is set to 'pixel'.
    image_roi : 4-tuple
        Defines the region of interest in the input images that ist loaded to
        the NLInputData. (left, right, top, bottom), all coordinates are
        inclusive!

        This can be used to reduce the memory consumption
        when loading multiple series of large images. If image_roi is None,
        the whole image is loaded.
    mode : str
        Sets the mode of loading. 'pixel' and 'region' can be used to implement
        different data handling, e.g. switchin slicing on/off. 'image' is meant
        as a bypass to the handling of averaging regions that returns whole
        image_roi. This is useful to access whole images without reimplementing
        the data access functions. It is used to do preceeding investigations
        to the nl-calibration.
    """

    avgreg_size: tuple
    mode = None
    _shape_new = None  # is this really required/helpful?
    _rois: list = []
    _img_roi: ImageRoi
    _allowed_modes: list = ["pixel", "region", "image"]

    bright_series: list = []
    dark_series: list = []

    def __init__(
        self,
        avgreg_size=None,
        image_roi=None,
        mode=None,
        **kwargs,
    ):
        # print("BASE", self.__class__)
        # print(avgreg_size, image_roi, mode)
        # print(kwargs)
        # print()


        if image_roi is None:
            image_roi = (None, None, None, None)
        if avgreg_size is None:
            avgreg_size = (1, 1)
        self.avgreg_size = tuple(avgreg_size)

        # FIXME: auslagern? später selbst initialisieren?
        # Hier wird schon auf Daten zugegriffen
        self._img_roi = ImageRoi(
            *self._resolve_roi_slice(image_roi, self.get_image_shape())
        )

        if mode is None:
            # depending on region size
            if self.avgreg_size[0] == 1 and self.avgreg_size[1] == 1:
                self.mode = "pixel"
            else:
                self.mode = "region"
            logger.debug("Set mode to '%s'.", self.mode)
        else:
            assert mode in self._allowed_modes
            self.mode = mode
        return

    # ---- API functions

    def read_bright_data(self):
        """Read bright data."""
        if self.bright_series is None or len(self.bright_series) == 0:
            raise Exception("No bright series set!")
        raise NotImplementedError()

    def read_dark_data(self):
        """Read dark data."""
        if self.dark_series is None or len(self.dark_series) == 0:
            raise Exception("No dark series set!")
        raise NotImplementedError()

    def get_image_shape(self):
        """Determine image shape of current dataset."""
        raise NotImplementedError()

    def set_bright_series(self, bright_series=None):
        """
        Set list of series names of bright data to load from dataset.

        The reason for this setter methods against setting self.bright_series
        directly, is to implement reasonable defaults in child classes, e.g.
        add all available series.

        Parameters
        ----------
        bright_series : list of str, optional
            Names of the series to load. The default is None which loads all
            available bright series.

        Returns
        -------
        None.

        """
        self.bright_series = bright_series
        return

    def set_dark_series(self, dark_series=None):
        """
        Set list of series names of dark data to load from dataset.

        The reason for this setter methods against setting self.dark_series
        directly, is to implement reasonable defaults in child classes, e.g.
        a fixen series name.

        Parameters
        ----------
        dark_series : list of str, optional
            Names of the series to load. The default is None which loads all
            available dark series.

        Returns
        -------
        None.

        """
        self.dark_series = dark_series
        return

    # ---- Slice functions

    # def set_rois_image_continuous(self, avgreg_size=None, img_shape=None):
    def set_rois_image_continuous(self):
        """
        Sets internal ROI list.

        ROIs are relative to self._img_roi.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        width, height = self.avgreg_size

        roi_w = np.abs(self._img_roi.end_x - self._img_roi.begin_x + 1)
        roi_h = np.abs(self._img_roi.end_y - self._img_roi.begin_y + 1)
        img_shape = (roi_h, roi_w)

        if (roi_h % height) != 0:
            logger.debug(
                "ROI-height (%d) is not multiple of height (%d)!", roi_h, height
            )
        if (roi_w % width) != 0:
            logger.debug("ROI-width (%d) is not multiple of width (%d)!", roi_w, width)

        self._rois, self._shape_new = self.get_rois_image_continuous(
            width, height, img_shape
        )
        return

    def get_rois_image_continuous(self, width, height, img_shape):
        """Create ROI lists.

        returns rois and shape of roi-image (reduced resolution image).
        The shape maybe helpful when reshaping roi-results to a
        result image.

        The splitting starts at image coordinate (0,0) to the other border
        in x/y direction as long as the whole roi fits onto the image.
        If the image width/heigth is not a multiple of width/height, the remaining
        pixels are ignored.

        Parameters
        ----------
        width, height : int
            horizontal and vertical size of regions
        image_shape : 2-tuple
            shape of image to be sliced
        Returns
        -------
        rois : list of ImageRoi
            ROIs
        shape_new : 2-tuple
            virtual size of roi-image. Can be used to reshape ROI-data to new image.

        """
        xrng = np.arange(
            0, img_shape[-1] + 1, width
        )  # negative index works with 2D images and 3D imagestacks
        yrng = np.arange(0, img_shape[-2] + 1, height)

        rois = []
        for i in range(1, len(yrng)):
            for j in range(1, len(xrng)):
                # print(
                #     "[{:d}:{:d}, {:d}:{:d}]".format(
                #         yrng[i - 1], yrng[i], xrng[j - 1], xrng[j]
                #     )
                # )
                rois.append(
                    ImageRoi(
                        begin_x=xrng[j - 1],
                        end_x=xrng[j] - 1,
                        begin_y=yrng[i - 1],
                        end_y=yrng[i] - 1,
                    )
                )

        shape_new = (len(yrng) - 1, len(xrng) - 1)
        return rois, shape_new

    def _resolve_roi_slice(self, roi, shape):
        img_h, img_w = shape

        # Anfänge auf 0 setzen, wenn None
        begin_x = 0 if roi[0] is None else roi[0]
        begin_y = 0 if roi[2] is None else roi[2]
        # Anfänge auf Ende - Wert setzen, wenn Wert negativ
        begin_x = img_w + begin_x if begin_x < 0 else begin_x
        begin_y = img_h + begin_y if begin_y < 0 else begin_y

        # Ende auf End setzen, wenn None. ODER End -1 ???
        end_x = img_w - 1 if roi[1] is None else roi[1]
        end_y = img_h - 1 if roi[3] is None else roi[3]
        # Ende auf End - Wert setzen, wenn Wert negativ
        end_x = img_w + end_x if end_x < 0 else end_x
        end_y = img_h + end_y if end_y < 0 else end_y

        # some checks for valid coordinates
        assert begin_x >= 0 and begin_x < img_w, "begin_x not in 0..img_width!"
        assert end_x >= 0 and end_x < img_w, "end_x not in 0..img_width!"
        assert begin_y >= 0 and begin_y < img_h, "begin_y not in 0..img_height!"
        assert end_y >= 0 and end_y < img_h, "end_x not in 0..img_height!"
        assert end_x >= begin_x, "begin_x needs to be smaller/equal than end_x"
        assert end_y >= begin_y, "begin_y needs to be smaller/equal than end_y"

        return begin_x, end_x, begin_y, end_y


# ------------------------- Storage Classes ----------------------------------

# pylint: disable=R0903

class NLInputAVGData(object):
    mean = None
    var = None
    ovrld = None
    n = None

class NLInputData(object):
    series_name = ""
    inttimes = None  # e.g. (100,)
    rois = None
    samples = None  # e.g. (100, 240816, 60)
    avgdat = None

class NLSelectedData(object):
    samples = None
    y0 = 0.0
    B0 = 0.0
    mean = None
    var = None
    inttimes = None
    series_name = ""
    roi_index = None
    roi = None


class NLResultData(object):
    inttimes = None
    ref_val = None
    ref_point = None
    y = None
    y_rel = None
    ref_fun = None
    nl_fun = None
    use_raw_counts = None
    outlier_limits = ((), (), ())

class PDNL_Correction():
    diffs = None
    inttimes = None
    rates = None

    def __init__(self, diffs, inttimes, rates ):
        self.diffs = diffs
        self.inttimes = inttimes
        self.rates = rates

    def correct(self, y, ti, max_iter=10):
        """
        - indizies in inttimes suchen, die um ti liegen
        - zwischen diesen spalten aus diffs interpolieren
          -> gibt 1D-array
        """
        t_idx = np.argwhere(self.inttimes < ti).T[0][-1]
        a = self.diffs[:, t_idx]
        b = self.diffs[:, t_idx+1]
        # diff_ti is 1D correction slice for inttime ti
        diff_ti = a + (b-a) * (ti-self.inttimes[t_idx]) / (self.inttimes[t_idx+1]-self.inttimes[t_idx])
        self.diff_ti = diff_ti

        y_curr = y
        diff_old = np.full_like(y, -np.inf)
        i = 0
        while True:
            i += 1
            B = y_curr / ti
            diff = np.interp(B, self.rates, diff_ti, left=np.nan, right=np.nan)
            y_curr = y + diff
            delta = diff - diff_old
            diff_old = diff
            # print("B: {!r} \ny: {!r} \ndiff: {!r} \ndelta: {!r}\n\n".format(B, y_curr, diff, delta))
            if np.all(delta < .5) or i >= 5:
                #print("Delta < .5!")
                break
            if i >= max_iter:
                print("Max iterations reached!")
                break
        #print(i)
        return y_curr


# simple data container, should be replaced by something meaningfull
# raise LazyProgrammerException()
class Data(object):
    pass


# @dataclass
# class DataSetInfo(object):
#     """Stores info about plotted datasets.

#     Can format the data to string.
#     Infoobjekt kann sich selbst für spätere Ausgbe formatieren. Man spart
#     sich eine externe Funktion im InfoPicker, die das Format kennt.
#     """

#     series: str
#     roi_idx: int
#     roi: ImageRoi

#     def __str__(self):

#         if self.roi.begin_x == self.roi.end_x:
#             sx = "x:{:d}".format(self.roi.begin_x)
#         else:
#             sx = "x:{:d}..{:d}".format(self.roi.begin_x, self.roi.end_x)
#         if self.roi.begin_y == self.roi.end_y:
#             sy = "y:{:d}".format(self.roi.begin_y)
#         else:
#             sy = "y:{:d}..{:d}".format(self.roi.begin_y, self.roi.end_y)
#         return 'Series: "{:s}" ROI: {:s} {:s}'.format(self.series, sx, sy)
