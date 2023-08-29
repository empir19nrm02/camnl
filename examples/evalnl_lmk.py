# -*- coding: utf-8 -*-
"""
@author: Christian Schrader
@email: christian.schrader@ptb.de
"""
import os
import sys
from camnl import NLCalibration
from camnl.nlplot import NLPlot
from camnl_data_lmk import ImportNLCalibData_lmk

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json


def main():
    global inlcd, nlcal, nlp, dat, dat_sel, y0, nlplt
    # return

    path_in = os.path.normpath("M:/example_nl-data")
    camlabel = "examplecam"

    groups_nl = None
    # groups_nl = [
    #     "open",
    # ]

    # Define main region of interrest of the image.
    # roi = (None, None, None, None)
    w = 10
    roi = (700 - w, 700 + w, 520 - w, 520 + w)

    inlcd = ImportNLCalibData_lmk(
        dirname=path_in,
        # input_mode='samples'
        # input_mode='averaged',
        avgreg_size=None,
        image_roi=roi,
    )
    inlcd.set_bright_series(groups_nl)
    inlcd.set_dark_series()
    inlcd.set_rois_image_continuous()

    print("IMG-ROI: %s" % (repr(inlcd._img_roi)))

    nlplt = NLPlot(savedir="plots", camlabel=camlabel)
    nlcal = NLCalibration(
        max_sensor_value=4095,
        spnl_refpoint=2000,
        nl_fun_kwparams={"order": 5},  # define parameters for nl-polynomial
    )

    # -----------------------------------------------------------------

    # different preliminary tests
    # eval_prelim_exam(inlcd, nlplt)

    # load data for nl-evaluation
    dat, dat_dark_short, y0, dat_sel = eval_load_data(inlcd, nlcal)

    # avaluate nl and plot result
    eval_plot_nl(nlplt, dat_sel)
    return


def eval_prelim_exam(inlcd, nlplt):
    nlplt.plot_sequence_diff_to_mean(inlcd)
    nlplt.plot_darksignal_over_inttime(inlcd)
    nlplt.plot_darksignal_stats(inlcd, threshold=10)
    return


def eval_load_data(inlcd, nlcal):
    dat_dark_short = inlcd.read_dark_data()
    # dat_dark_long = inlcd.read_dark_data('dark_long')
    y0 = nlcal.calc_y0(dat_dark_short)

    dat = inlcd.read_bright_data()
    # ti_min wegen Sprung bei 270us
    dat_sel = nlcal.select_nl_data(dat, inttime_limits=(0.0003, None), y0=y0)
    return dat, dat_dark_short, y0, dat_sel


def eval_plot_nl(nlplt, dat_sel):
    global nldat

    figname = "Non-Linearity"
    # background
    nldat = nlcal.calc_sigproc_nl(dat_sel, ref_point=2000, mode="mean")

    ti_max = np.max([d.inttimes[-1] for d in dat_sel[0]])

    fig = nlplt.plot_sigproc_nl(
        nldat,
        plot_data=True,
        use_it_colors=False,
        plot_nl=True,
        fig=figname,
        filename="non-linearity",  # save after last plot,
    )

    # ax = fig.gca()
    # ax.set_ylim(0.95, 1.08)
    # nlplt.save_figure(fig, fn + "_zoom")

    return


# --- main call
if __name__ == "__main__":
    main()
