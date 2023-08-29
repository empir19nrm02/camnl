# -*- coding: utf-8 -*-
"""
@author: Christian Schrader
@email: christian.schrader@ptb.de
"""
import os
import logging
import numpy as np
import gzip
import bz2

import sys
import time

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    # format='%(name)s (%(lineno)d) %(levelname)s - %(message)s',
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("lmk_image")
logger.disabled = True


class lmk_image(object):
    """Import of TechnoTeam LMK images."""

    _picture = None
    _header = {}

    def __init__(self, fn=None):
        self._header = {}
        if fn is not None:
            self._picture = self.load_file(fn)
        return

    def get(self, index=None):
        if len(self._picture) == 1:
            #  for monochrome images, return only this
            return self._picture[0]
        elif index == None:
            # return all colors
            return self._picture
        else:
            # index: 0:Blue, 1:Green, 2:Red
            return self._picture[index]

    def get_header(self, name=None):
        if name:
            return self._header[name]
        return self._header

    def __repr__(self):
        return repr(self._picture)

    ########################################################################
    #  Text-Files
    ########################################################################

    def _handle_TextFormat(self, f, dtype=float, colors=1):
        logger.debug('_handle_TextFormat')
        f.seek(0)
        f.readline()  # Typ-Header verwerfen
        # 2nd line contains first/last column/row
        fLine, lLine, fCol, lCol = np.asarray(
            f.readline().rsplit(), dtype=np.int32
        )  # dimensions
        pic_w = lCol - fCol + 1
        pic_h = lLine - fLine + 1
        self.header = {
            "Lines": pic_h,
            "Columns": pic_w,
            "FirstLine": fLine,
            "FirstColumn": fCol,
        }

        # loadtxt um ca. Faktor 10 schneller als fromfile()!
        pic = np.loadtxt(f, dtype=dtype)
        print(pic)
        self._pic = pic
        # pic = pic.flatten() # dann ist das Format das gleiche wie binär

        pic2 = np.zeros((colors, pic_h, pic_w))
        for c in range(colors):
            pic2[c] = pic[c::colors].reshape((pic_h, pic_w))

        return pic2

        return pic.reshape((colors, pic_h, pic_w))

    def _handle_TextCamera(self, f):
        logger.debug("_handle_TextCamera")
        return self._handle_TextFormat(
            f, dtype=np.uint16
        )  # FIXME: data type fixed?

    def _handle_TextLuminance(self, f):
        logger.debug("_handle_TextLuminance")
        return self._handle_TextFormat(f, dtype=float)

    def _handle_TextColor(self, f):
        logger.debug("_handle_TextColor")
        return self._handle_TextFormat(f, dtype=float, colors=3)

    ########################################################################
    #  Binary Files
    ########################################################################

    def _handle_BinaryFormat(self, f, dtype=np.float32, colors=1):
        # FIXME: convert headers to correct data types?
        logger.debug("_handle_BinaryFormat")
        # tic = time.perf_counter()
        f.seek(0)
        self._header = {}
        for line in iter(f.readline, b"\r\n"):
            v = line.decode().strip().split("=", 1)
            if v[0][0] == "|":
                v[0] = v[0][
                    1:
                ]  # discard | in header keys. FIXME: can this lead to problems?
            self._header[v[0]] = v[1]
        # print("dt h", time.perf_counter()-tic)
        f.seek(f.tell() + 1)  # skip 0-byte

        pic_h = int(self._header["Lines"])
        pic_w = int(self._header["Columns"])
        # tic = time.perf_counter()

        pic = np.fromfile(f, dtype)
        self._pic = pic
        self._shape = (pic_h, pic_w)

        # print("nbytes", dtype().nbytes)
        # print("pic_w*pic_h*colors * nbytes", pic_w*pic_h*colors*dtype().nbytes)
        # print("pic", len(pic))

        # FIXME: other reordering for color images? Was not really used.
        pic2 = np.zeros((colors, pic_h, pic_w), dtype=dtype)
        for c in range(colors):
            pic2[c] = pic[c::colors].reshape((pic_h, pic_w))

        # print("dt d", time.perf_counter()-tic)
        return pic2

    # def _handle_BinaryFormat_old(self, f, dtype=np.float32, colors=1):
    #     # slower version, reads the file twice :/
    #     # logger.debug('_handle_BinaryFormat')
    #     # tic = time.perf_counter()
    #     f.seek(0)
    #     data_s = f.read()
    #     header_end = data_s.find(b"\x00")  # binär-daten
    #     if header_end == -1:
    #         raise Exception("Header error!")
    #     header = data_s[:header_end].decode()
    #     self.header = header
    #     # print("dt", time.perf_counter()-tic)
    #     data_s = data_s[header_end + 1 :]  # Nullbyte verwerfen
    #     # print("dt", time.perf_counter()-tic)

    #     for row in header.splitlines():  # Header aufteilen
    #         v = row.strip().split("=", 1)
    #         if len(v[0]) == 0:
    #             continue
    #         if v[0][0] == "|":
    #             v[0] = v[0][1:]  # | verwerfen
    #         self._header[v[0]] = v[1]
    #     # FIXME: convert headers to correct data types?

    #     pic_h = int(self._header["Lines"])
    #     pic_w = int(self._header["Columns"])
    #     tic = time.perf_counter()
    #     pic = np.fromstring(data_s, dtype)
    #     self._pic = pic
    #     self._shape = (pic_h, pic_w)
    #     # print("dt", time.perf_counter()-tic)
    #     # print()
    #     # print("data_s", len(data_s))
    #     # print("nbytes", dtype().nbytes)
    #     # print("pic_w*pic_h*colors * nbytes", pic_w*pic_h*colors*dtype().nbytes)
    #     # print("pic", len(pic))

    #     # FIXME: other reordering for color images? Was not used.
    #     pic2 = np.zeros((colors, pic_h, pic_w), dtype=dtype)
    #     for c in range(colors):
    #         pic2[c] = pic[c::colors].reshape((pic_h, pic_w))

    #     return pic2

    def _handle_Camera(self, f):
        logger.debug("_handle_BinaryCamera")
        return self._handle_BinaryFormat(f, dtype=np.uint16)

    def _handle_Luminance(self, f):
        logger.debug("_handle_BinaryLuminance")
        return self._handle_BinaryFormat(f, dtype=np.float32)

    def _handle_Color(self, f):
        logger.debug("_handle_BinaryColor")
        return self._handle_BinaryFormat(f, dtype=np.float32, colors=3)

    def load_file(self, fn):
        handler = {
            "ushort": self._handle_TextCamera,
            "float": self._handle_TextLuminance,
            "rgbfloat": self._handle_TextColor,
            "Pic98::TPlane<unsigned short>": self._handle_Camera,
            "Pic98::TPlane<float>": self._handle_Luminance,
            "Pic98::TPlane<Pic98::TRGBFloatPixel>": self._handle_Color,
        }

        if os.path.isfile(fn + ".gz"):
            open_func = gzip.open
            fn_full = fn + ".gz"
        elif os.path.isfile(fn + ".bz2"):
            open_func = bz2.BZ2File
            fn_full = fn + ".bz2"
        else:
            open_func = open
            fn_full = fn

        with open_func(fn_full, "rb") as f:
            # extract first line
            # text files contain only "ushort", "float", "rgbfloat"
            # binary files contain header with "Typ= ..."
            # -> see keys in handler-dict
            picType = f.readline().strip().decode().split("=", 1)[-1]
            f.seek(0)  # reset
            if picType not in handler:
                raise Exception("No TT-LMK-file: " + fn_full)
            self._picture = handler[picType](f)
            return self._picture

        raise Exception("Could not open '%s'" % (fn_full))
        return
