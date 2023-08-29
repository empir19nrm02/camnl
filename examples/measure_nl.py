# -*- coding: utf-8 -*-
"""
Example script to measure non-linearity by variation of integration time with
TechnoTeam ILMDs.

@author: Christian Schrader
@email: christian.schrader@ptb.de

Use libtiff to save compressed tiff images.
Requires:
    https://pypi.org/project/libtiff/
    https://pypi.org/project/bitarray/
"""

import os
import numpy as np
import tempfile
import datetime
from collections import OrderedDict
import json
from lmk import LMK, LMK_Error
from lmk_image import lmk_image
import logging
import PIL
from libtiff import TIFF, TIFFfile, TIFFimage

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("libtiff").setLevel(logging.WARNING)

opjoin = os.path.join  # to make lines shorter


def main():
    global cam  # keep obkect available at python console after run

    cam = LMK_Task()
    cam.cam_init("CAMERANAME", "LENSNAME")  # Adjust to camera and lens!
    cam.measure_nl(name="examplecam", ti_end=0.008, n_img=25, ti_steps=50)
    cam.lmk.Close()
    return cam


class LMK_Task:
    lmk = None
    data_dir = "data"  # subdirectory where to store image data

    def cam_init(self, camname, lensname):
        # open camera and set acquisition settings
        lmk = LMK()
        try:
            lmk.Close()
        except LMK_Error:
            pass
        lmk.Open()
        lmk.SetNewCamera2(camname, lensname)
        lmk.SetFilterWheel(4)  # VL
        self.lmk = lmk
        return

    def measure_nl(
        self,
        name=None,
        series="open",  # names subseries with different nd-filter
        n_img=5,
        ti_start=None,
        ti_end=None,
        ti_steps=25,
        store_samples=True,  # store each sample image
        store_avg=False,  # store only mean and variance of all sample images
    ):
        """
        Measure nl-data by variation of integration time.

        By default store each sample image individually. For this the evaluation
        of the nl is implemented.
        Alternatively averaged images can be saved. Each dataset consists of:
            - mean, variance
            - number of overload pixels : with this can be tested of in any of the
              sample images an overload occured. Otherwise this would not be visible
              until all samples are overloaded. For partial overload only the mean value
              would be shifted which would result in a determined non-linearity that
              is not real.
        """
        assert store_samples or store_avg, "No storage method selected!"

        if self.lmk is None:
            raise Exception("Camera not initialized!")

        if name is None:
            name = datetime.datetime.now().strftime("%Y-%m-%dT%H.%M.%S")

        # Storage outside of data_dir is prohibited. It is easy to destroy data.
        data_dir_meas = opjoin(self.data_dir, name, series)
        print("Basedir: ", data_dir_meas)
        os.makedirs(data_dir_meas, exist_ok=False)

        # determine integration times
        _, _, _, cam_ti_min, cam_ti_max = self.lmk.GetIntegrationTime()
        if ti_end is None:
            ti_end = cam_ti_max
            print("Use default max. integration time: {:f} s".format(ti_end))

        if ti_start is None:
            ti_start = cam_ti_min
            print("Use cam's min. integration time: {:f} s".format(ti_start))

        int_time_list = np.linspace(ti_start, ti_end, num=ti_steps, endpoint=True)

        # transform list to inttimes that can be realized
        # takes some time but makes debugging easier if only a few times remain
        int_time_list = [self.lmk.SetIntegrationTime(ti) for ti in int_time_list]

        int_times_real = OrderedDict()
        with tempfile.TemporaryDirectory() as tmpdir:
            n_all_pts = len(int_time_list)
            for nr_point, int_time in enumerate(int_time_list):
                int_time_real = self.lmk.SetIntegrationTime(int_time)
                int_times_real[nr_point] = int_time_real
                print(
                    "\n{:d}/{:d} IntTime: {:f} IntTimeReal: {:f}".format(
                        nr_point + 1, n_all_pts, int_time, int_time_real
                    )
                )

                data_dir_curr = tmpdir
                if store_samples:
                    # create directory for all samples of each integration time
                    data_dir_curr = opjoin(data_dir_meas, str(nr_point))
                    os.makedirs(data_dir_curr, exist_ok=False)
                if store_avg:
                    stats = PtPStats(event_fun=lambda x: (x >= 4095))
                    self.stats = stats  # debug

                for i in range(n_img):
                    if store_samples:
                        img_path = os.path.abspath(
                            opjoin(data_dir_curr, "{:d}.pus".format(i))
                        )
                    else:
                        img_path = os.path.abspath(opjoin(data_dir_curr, "tmpimg.pus"))
                    print(".", end="")
                    self.lmk.Grab()
                    # image type needs to fit the filename ending!
                    self.lmk.SaveImage(-3, img_path)
                    lmkimg = lmk_image(img_path)
                    if store_avg:
                        stats(lmkimg.get())

                # with open(opjoin(data_dir_meas, "inttimes.dat"), mode="a") as f:
                #     f.write("{:d}\t{:.8f}\n".format(nr_point, int_time_real))

                if store_avg:
                    ## PIL destroys images when using compression :/
                    # tmpimg = PIL.Image.fromarray(stats.mean.astype(np.float32))
                    # tmpimg.save(
                    #     opjoin(data_dir_meas, "{:d}_mean.tiff".format(nr_point)),
                    # )
                    # tmpimg = PIL.Image.fromarray(stats.var.astype(np.float32))
                    # tmpimg.save(
                    #     opjoin(data_dir_meas, "{:d}_var.tiff".format(nr_point)),
                    #     compression=tiff_compression,
                    # )

                    tmpimg = TIFFimage(stats.mean.astype(np.float32), description="")
                    tmpimg.write_file(
                        opjoin(data_dir_meas, "{:d}_mean.tiff".format(nr_point)),
                        compression="lzw",
                        verbose=False,
                    )

                    tmpimg = TIFFimage(stats.var.astype(np.float32), description="")
                    tmpimg.write_file(
                        opjoin(data_dir_meas, "{:d}_var.tiff".format(nr_point)),
                        compression="lzw",
                        verbose=False,
                    )

                    # save overload-counter
                    # store only if values > 0 exist
                    if stats.event_fun is not None:
                        event_count = stats.event_count
                        if np.any(event_count > 0):
                            # reduce data type to smalles possible one
                            dtypes = [np.uint8, np.uint16, np.uint32, np.uint32]
                            maxval = event_count.max()
                            dtype_idx = 0
                            if maxval > 0:
                                dtype_idx = int(np.log2(event_count.max()) // 8)
                            event_count = event_count.astype(dtypes[dtype_idx])

                            tmpimg = TIFFimage(event_count)
                            tmpimg.write_file(
                                opjoin(
                                    data_dir_meas, "{:d}_ovrld.tiff".format(nr_point)
                                ),
                                compression="lzw",
                                verbose=False,
                            )

                    header_s = ""
                    for k, v in lmkimg.get_header().items():
                        header_s += "{!s}={!s}\n".format(k, v)
                    with open(
                        opjoin(data_dir_meas, "{:d}_hdr.dat".format(nr_point)),
                        mode="w",
                    ) as f:
                        f.write(header_s)

            meas_data = OrderedDict()
            meas_data["nr_samples"] = n_img
            meas_data["inttimes"] = int_times_real
            self.meas_data = meas_data
            with open(opjoin(data_dir_meas, "measurement.json"), mode="w") as f:
                json.dump(meas_data, f, indent=2)
        return

    def test_equivalence(self, name, series=None):
        """
        Test equivalence of ptp-results and direct averaging.

        Implicit testing of correct/lossless tiff storage.
        """
        # find available series of inttime-variation
        series_list = []
        if series is None:
            data_dir = opjoin(self.data_dir, name)
            for fn in os.listdir(data_dir):
                fn_dir = opjoin(data_dir, fn)
                if os.path.isdir(fn_dir):
                    series_list.append(fn)
        else:
            series_list = [series]  # default

        for series in series_list:
            data_dir_meas = opjoin(self.data_dir, name, series)

            for fn in os.listdir(data_dir_meas):
                fn_dir = opjoin(data_dir_meas, fn)
                fn_mean = fn_dir + "_mean.tiff"
                fn_var = fn_dir + "_var.tiff"
                if (
                    os.path.isdir(fn_dir)
                    and os.path.exists(fn_mean)
                    and os.path.exists(fn_var)
                ):
                    print()
                    print(fn, ":")
                    print(fn_dir)
                    print(fn_mean)
                    print(fn_var)

                    imgstack = []

                    for fn_img in os.listdir(fn_dir):
                        fnpath_tmp = os.path.normpath(opjoin(fn_dir, fn_img))
                        imgstack.append(lmk_image(fnpath_tmp).get())
                    imgstack = np.asarray(imgstack)
                    mean_samples = imgstack.mean(axis=0)
                    var_samples = imgstack.var(axis=0, ddof=1)
                    mean_img = np.asarray(PIL.Image.open(fn_mean))
                    var_img = np.asarray(PIL.Image.open(fn_var))
                    print("Equiv. Mean:", np.allclose(mean_samples, mean_img))
                    print("Equiv. Var.:", np.allclose(var_samples, var_img))
        return


class PtPStats:
    """
    Calculate Point-to-Point-Mean and -Variance.

    Input data can be scalar or multidimensional ndarrays.

    Event function can be used to count overload occurences.
    """

    sum_x = None
    sum_x2 = None
    mean = None
    n = 0
    var = None
    event_fun = None

    def __init__(self, event_fun=None):
        self.event_fun = event_fun

    def __call__(self, x, init=False):
        if self.sum_x is None or init:
            # init
            self.n = 1
            self.sum_x = np.asarray(x, dtype=np.float64)
            self.sum_x2 = self.sum_x**2
            self.mean = self.sum_x
            # nans require float
            var = np.full_like(x, np.nan, dtype=np.float64)
            if self.event_fun is not None:
                self.event_count = np.zeros_like(x, dtype=int)
            return self.mean, var

        if self.event_fun is not None:
            # eval event_fun on original input data
            self.event_count += np.asarray(self.event_fun(x), dtype=int)

        x = np.asarray(x, dtype=np.float64)
        n0 = self.n
        self.n += 1
        self.mean = self.mean + (x - self.mean) / self.n  # dont use +=
        self.sum_x = self.sum_x + x
        self.sum_x2 = self.sum_x2 + (x**2)
        # store to self to make it readable again
        self.var = (self.sum_x2 - (self.sum_x**2) / self.n) / n0
        return self.mean, self.var

    @property
    def std(self):
        if self.var is not None:
            return np.sqrt(self.var)
        return None

    def reset(self):
        self.sum_x = None
        self.var = None
        self.mean = None
        return


if __name__ == "__main__":
    main()
    pass
