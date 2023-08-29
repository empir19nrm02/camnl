# -*- coding: utf-8 -*-
"""
@author: Christian Schrader
@email: christian.schrader@ptb.de
"""

import os
import sys
import numpy as np
import numpy.polynomial.polynomial as po

# from collections import namedtuple

from cycler import cycler

# from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib as mpl

from .infopicker import InfoPicker

# from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams["image.interpolation"] = "none"
plt.rc(
    "axes",
    prop_cycle=(
        cycler(
            "color",
            [
                "red",
                "green",
                "blue",
                "orange",
                "lightgray",
                "magenta",
                "cyan",
                "black",
                "darkred",
                "springgreen",
            ],
        )
    ),
)


import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    # format='%(name)s (%(lineno)d) %(levelname)s - %(message)s',
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("nlplot")
logger_picker = logging.getLogger("InfoPicker")


"""
1. plot_sequence_diff_to_mean(nlc) : feststellen, ob Sequenzen kontinuierlich
   sind oder ob Sprünge auftreten.

"""


class NLPlot():

    savedir = None
    camlabel = None

    def __init__(self, savedir="plots", camlabel=None, figsize=(6.4, 4.8)):
        self.savedir = savedir
        if camlabel is None:
            camlabel = "unlabled_camera"
        self.camlabel = camlabel
        self.figsize = figsize
        return

    def plot_sequence_diff_to_mean(
        self,
        inlcd,
        overload_value=4095,
        dark=False,
    ):
        """
        Plots the difference of the mean of each image to the mean of the series.

        For a meaningful determination of non-linearity the internal timing of the
        camera is fundamental. If this is given, the plotted lines are curved but
        continous, without steps. If steps occur at distinct integration times,
        the data can not be used to calculate a common non-linearity function for
        all integration time regions.

        This function plots the mean of each image of a series to the expected value
        of the whole series at the corresponding integration time.

            - all images of a series are read in
            - the last integration time where no overload occurs is determined
            - the average count rate is determined by linear regression
            - the differenz of each images mean value to the expected value
              (average_count_rate * integration_time) is plotted.

        Parameters
        ----------
        inlcd : ImportNLCalibData object
            The object needs to be fully configured.

        Returns
        -------
        None.

        """

        T_all = []
        Y_all = []
        Y_m_all = []
        Label_all = []

        if dark:
            series_list = inlcd.get_dark_series()
            prefix = "dark"  # set prefix to not overwrite bright plots
        else:
            series_list = inlcd.get_bright_series_available()
            prefix = ""

        for i_series, series in enumerate(series_list):
            print(series)
            imgdat = inlcd.read_inttime_series(series, mode="image")
            img = imgdat.samples
            print(img.shape)
            t = imgdat.inttimes

            # average over whole image and repetitive measurements
            m = img.mean(axis=(1, 2, 3))
            # overload ratio, 0-1
            o = np.sum(img == overload_value, axis=(1, 2, 3)) / np.prod(img.shape[1:])

            """
            Do not test for 0 pixel in overload but if it are less then 10%.
            Otherwise hot pixels can interfere.
            """
            # idx_valid = o[0]==0
            idx_valid = o < 0.1

            T = t[idx_valid]
            Y = m[idx_valid]
            lincoeff = np.polyfit(T, Y, 1)
            Y_m = np.polyval(lincoeff, T)  # expected average count value

            title = "Diff to Mean - Cam: {:s} Series: {:s}".format(
                self.camlabel, series
            )
            fn = "diff_to_mean"
            if prefix:  # > ""
                fn = fn + "_" + prefix

            # do not use series label as file name. They can contain characters that
            # are forbidden in filenames
            fn_i = fn + "_%d" % i_series

            fig = plt.figure(
                "Diff to Mean over Inttime: " + series, figsize=self.figsize
            )
            fig.clear()
            ax = fig.gca()
            ax.plot(T, Y - Y_m, ".-", label=series)
            ax.set_xlabel("$t_\mathrm{i}$ / s")
            ax.set_ylabel("$y - \overline{y}$ / DN")
            ax.grid(True)
            ax.legend()
            #ax.set_title(title)
            fig.set_tight_layout(True)
            self.save_figure(fig, fn_i + "_t")

            fig2 = plt.figure(
                "Diff to Mean over Counts: " + series, figsize=self.figsize
            )
            fig2.clear()
            ax2 = fig2.gca()
            ax2.plot(Y, Y - Y_m, ".-", label=series)
            ax2.set_xlabel("$y$ / counts")
            ax2.set_ylabel("$y - \overline{y}$ / DN")
            ax2.grid(True)
            ax2.legend()
            #ax2.set_title(title)
            fig2.set_tight_layout(True)
            self.save_figure(fig2, fn_i + "_c")

            T_all.append(T)
            Y_all.append(Y)
            Y_m_all.append(Y_m)
            Label_all.append(series)
            # print("Serie: %s - Ti_max: %.6f" %(series, T[-1]))

        fig_all_t = plt.figure(
            "Diff to Mean over Inttime: All Series", figsize=self.figsize
        )
        fig_all_t.clear()
        ax_all_t = fig_all_t.gca()
        fig_all_c = plt.figure(
            "Diff to Mean over Counts: All Series", figsize=self.figsize
        )
        fig_all_c.clear()
        ax_all_c = fig_all_c.gca()

        #ax_all_t.set_title("Diff to Mean - Cam: {:s}".format(self.camlabel))
        #ax_all_c.set_title("Diff to Mean - Cam: {:s}".format(self.camlabel))

        # # sort plots by integration time
        sortidx = np.argsort([x[-1] for x in T_all])
        # T_all = [T_all[i] for i in sortidx]
        # Y_all = [Y_all[i] for i in sortidx]
        # Y_m_all = [Y_m_all[i] for i in sortidx]

        for i in sortidx:
            ax_all_t.plot(T_all[i], Y_all[i] - Y_m_all[i], ".-", label=Label_all[i])
            ax_all_c.plot(Y_all[i], Y_all[i] - Y_m_all[i], ".-", label=Label_all[i])

        ax_all_t.legend()
        ax_all_t.set_xlabel("$t_\mathrm{i}$ / s")
        ax_all_t.set_ylabel("$y - \overline{y}$ / DN")
        ax_all_t.grid(True)
        fig_all_t.set_tight_layout(True)
        self.save_figure(fig_all_t, fn + "_all_t")

        ax_all_c.legend()
        ax_all_c.set_xlabel("$y$ / DN")
        ax_all_c.set_ylabel("$y - \overline{y}$ / DN")
        ax_all_c.grid()
        fig_all_c.set_tight_layout(True)
        self.save_figure(fig_all_c, fn + "_all_c")

        return

    # def plot_luminance_over_inttime(self, inlcd, series_list=None):

    #     T_all = []
    #     Y_all = []
    #     Label_all = []

    #     figsize = self.figsize

    #     if series_list is None:
    #         print("verwende alle Serien")
    #         series_list = inlcd.get_bright_series_available()
    #     shape = inlcd.get_image_shape()
    #     print("ACHTUNG: fixes Auswertefenster!")
    #     h, w = shape
    #     cx = w // 2
    #     cy = h // 2
    #     # sl_y = slice(0,20) # oberer Rand
    #     sl_y = slice(cy - 10, cy + 10)  # mitte y
    #     # sl_y = slice(h-20,h) # unterer Rand
    #     #
    #     # sl_x = slice(0,20) # linker Rand
    #     sl_x = slice(cx - 10, cx + 10)  # mitte x
    #     # sl_x = slice(w-20,w) # rechter Rand

    #     print("Shape:", shape, sl_y, sl_x)
    #     for i_series, series in enumerate(series_list):
    #         m, v, t, o = inlcd.read_inttime_series(series, mode="avgpixel")
    #         print("First IntTimes:", t[0:5])  #
    #         # for im in m.T:
    #         #    foo = im.reshape(shape)[sl_y, sl_x]
    #         #    print("IMS", im.shape, foo.shape)
    #         imgs = np.asarray([im.reshape(shape)[sl_y, sl_x] for im in m.T])
    #         #            plt.plot(t, imgs.mean(axis=(1,2)), '.-')
    #         T = t
    #         Y = imgs.mean(axis=(1, 2))

    #         title = "Luminance over $t_\mathrm{i}$"
    #         title += " - Cam: {:s} Series: {:s}".format(self.camlabel, series)
    #         fn = "luminance_over_ti"

    #         fn_i = (
    #             fn + "_%d" % i_series
    #         )  # die Serien-Label erstmal nicht als Dateinamen nehmen. Nur fortlaufende Nummer. Seriennamen können unerlaubte Zeichen enthalten.

    #         fig = plt.figure("Luminance over Inttime: " + series, figsize=figsize)
    #         fig.clear()
    #         ax = fig.gca()
    #         ax.plot(T, Y, ".-", label=series)
    #         ax.set_xlabel("$t_\mathrm{i}$ / s")
    #         ax.set_ylabel("$L$ / cd/m²")
    #         ax.grid(True)
    #         ax.legend()
    #         ax.set_title(title)
    #         fig.set_tight_layout(True)
    #         self.save_figure(fig, fn_i)

    #         T_all.append(T)
    #         Y_all.append(Y)
    #         Label_all.append(series)

    #     fig_all = plt.figure("Luminance over Inttime: All Series", figsize=figsize)
    #     fig_all.clear()
    #     ax_all = fig_all.gca()

    #     ax_all.set_title("Luminance over Inttime - Cam: {:s}".format(self.camlabel))

    #     # Plots in aufsteigender maximaler Integrationszeit sortieren
    #     sortidx = np.argsort([x[-1] for x in T_all])

    #     for i in sortidx:
    #         Y_rel = (Y_all[i] - Y_all[i][-1]) / Y_all[i][-1]
    #         T_rel = T_all[i]  # / T_all[i][-1]
    #         # ax_all.plot(T_all[i], Y_all[i], '.-', label=Label_all[i])
    #         ax_all.plot(T_rel, Y_rel, ".-", label=Label_all[i])

    #     ax_all.legend()
    #     ax_all.set_xlabel("$t_\mathrm{i,rel}$ / s")
    #     ax_all.set_ylabel("$L$ / cd/m²")
    #     ax_all.grid(True)
    #     fig_all.set_tight_layout(True)
    #     self.save_figure(fig_all, fn + "_all")
    #     return

    def plot_darksignal_over_inttime(self, inlcd):
        fn = "darksignal_over_ti"

        figsize = self.figsize

        dark_s = dark_l = None
        dark_s = inlcd.read_inttime_series("dark_short", mode="image")
        dark_l = inlcd.read_inttime_series("dark_long", mode="image")

        mean_s = dark_s.samples.mean(axis=(1, 2, 3))
        mean_l = dark_l.samples.mean(axis=(1, 2, 3))

        fig_s = plt.figure("Dark short", figsize=figsize)
        fig_s.clear()
        ax_s = fig_s.gca()
        ax_s.plot(dark_s.inttimes, mean_s, "-", c="gray", alpha=0.1)
        lines_short = ax_s.plot(dark_s.inttimes, mean_s, ".", c="r")
        ax_s.set_xlabel("$t_\mathrm{i}$ / s")
        ax_s.set_ylabel("$y$ / DN")
        ax_s.set_title("Dark signal - short inttime")
        ax_s.legend([lines_short[0]], ["short"])
        ax_s.grid(True)
        fig_s.set_tight_layout(True)
        self.save_figure(fig_s, fn + "_short")

        # dark_l = getattr(nlc, "dark_long", None)
        if dark_l is not None:
            fig_l = plt.figure("Dark long", figsize=figsize)
            fig_l.clear()
            ax_l = fig_l.gca()

            lines_short = ax_l.plot(dark_s.inttimes, mean_s, ".", c="r")
            ax_l.plot(dark_l.inttimes, mean_l, "-", c="gray", alpha=0.1)
            lines_long = ax_l.plot(dark_l.inttimes, mean_l, ".", c="b")

            ax_l.set_xlabel("$t_\mathrm{i}$ / s")
            ax_l.set_ylabel("$y$ / DN")
            ax_l.set_title("Dark signal - long inttime")
            ax_l.legend([lines_short[0], lines_long[0]], ["short", "long"], loc="best")
            ax_l.grid(True)
            fig_l.set_tight_layout(True)
            self.save_figure(fig_l, fn + "_long")

        return

    def plot_darksignal_stats(self, nlc, threshold=10):

        # FIXME: das ist anhängig von Speicherart. Besser
        # gesamte Integrationszeitserie  einlesen und nur erstes Bild verwenden
        # FIXME: only dark_short? or all in nlc.dark_series?
        dat = nlc.read_inttime_series("dark_short", mode="image")

        # use only images with shortest integration time
        im = dat.samples[0]
        # im_m = im_0.mean(axis=0) # averaging along samples
        self.dat = dat

        im_min = im.min()
        im_max = im.max()
        print()
        print("Number of images:", len(im))
        print("Minimum value:", im_min)
        print("Maximum value:", im_max)

        fig_hist = plt.figure("Histogramm of Dark Signal", figsize=self.figsize)
        fig_hist.clear()
        ax_hist = fig_hist.gca()
        ax_hist.set_xlabel("$y$ / DN")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Histogramm of Dark Signal")

        ax_hist.hist(im.ravel(), bins=im_max - im_min, histtype="bar")
        fig_hist.set_tight_layout(True)
        self.save_figure(fig_hist, "darksignal_histogram")

        n_low = np.sum(im < threshold)
        print(
            "Number of pixels below threshold::",
            n_low,
            "(",
            n_low / len(im),
            ")",
        )
        print(
            "Percentage of pixels below threshold: %.3f %%" % (n_low / im.size * 100)  #
        )

        fig1 = plt.figure("Dark Image Single", figsize=self.figsize)
        fig1.clear()
        ax1 = fig1.gca()
        ax1.set_xlabel("pixel x")
        ax1.set_ylabel("pixel y")
        ax1.set_title("Dark Image Single")
        ax1.imshow(im[0])
        fig1.set_tight_layout(True)
        self.save_figure(fig1, "darkimage_single")

        fig2 = plt.figure("Dark Image Averaged", figsize=self.figsize)
        fig2.clear()
        ax2 = fig2.gca()
        ax2.set_xlabel("pixel x")
        ax2.set_ylabel("pixel y")
        ax2.set_title("Dark Image Averaged")
        ax2.imshow(im.mean(axis=0))
        fig2.set_tight_layout(True)
        self.save_figure(fig2, "darkimage_averaged")

        # entlang der Wiederholungsbilder ORen, d.h., wenn irgendwann mal
        # unter Schwellwert, dann wird der Pixel als unterhalb gewertet.
        im_low = im < threshold
        im_mask_low = np.full_like(im_low[0], False)
        for i in im_low:
            im_masked_low = np.logical_or(im_mask_low, i)

        fig3 = plt.figure("Below Threshold Image", figsize=self.figsize)
        fig3.clear()
        ax3 = fig3.gca()
        ax3.set_xlabel("pixel x")
        ax3.set_ylabel("pixel y")
        ax3.set_title("Below Threshold Image")
        ax3.spy(im_masked_low)
        fig3.set_tight_layout(True)
        self.save_figure(fig3, "below_threshold_image")

        # # Dunkelsignal-Mean und Standardabweichung über ti
        # noise = []
        # mean = []
        # ti_a = []
        # for i in range(25):
        #     im, ti = nlc.read_rep_images_inttime(group_name='dark_long/%d' % i)
        #     ap = im[:, 525-100:525+100,700-100:700+100]
        #     noise.append(np.std(ap, ddof=1))
        #     mean.append(np.mean(ap))
        #     ti_a.append(ti)
        # mean = np.asarray(mean)
        # plt.figure("noise")
        # plt.plot(ti_a, mean-mean[0])
        # plt.plot(ti_a, noise)

        return

    def save_figure(self, fig, fn, savedir=None, *args, **kwargs):
        if savedir is None:
            savedir = self.savedir
            if self.camlabel not in [None, ""]:
                savedir = os.path.join(savedir, self.camlabel)

        if os.path.exists(savedir):
            if not os.path.isdir(savedir):
                raise Exception("File with directory name exists!")
        else:
            os.makedirs(savedir)
        fpath = os.path.join(savedir, fn)
        fig.savefig(fpath, *args, **kwargs)
        return

    def plot_sigproc_nl(
        self,
        nldat,
        #
        plot_data=False,
        plot_data_param={},
        data_color="gray",
        #
        plot_nl=False,
        plot_nl_param={},
        nl_color="black",
        nl_label=None,
        #
        use_it_colors=None,
        it_corr=None,
        fig=None,
        title=None,
        figsize=None,
        # savedir=None,
        # camlabel=None,
        pickradius=0,
        filename="",
    ):

        # default plotting styles
        if len(plot_data_param) == 0:
            plot_data_param = {'ls': 'none', 'marker': '.'}
        if len(plot_nl_param) == 0:
            plot_nl_param = {'ls': '-', 'lw': 2}

        try:
            figsize = self.figsize
            if fig is None:
                fig = plt.figure("SigProc Nonlinearity")
                fig.clear()
            elif type(fig) == str:
                fig = plt.figure(fig, figsize=self.figsize)
            ax = fig.gca()

            use_picker = pickradius > 0

            # self.dbg_data_colors = []  # debug
            # self.dbg_c_idx = []

            # create palette
            if use_it_colors not in [0, None, False]:
                c_range = (
                    100  # int(use_it_colors) + 1 # Anzahl der zu verwendenden Farben
                )
                ti_colors = plt.cm.jet(np.linspace(0.1, 0.9, c_range))[:, 0:3]
                # self.dbg_ti_colors = ti_colors
                # Plan B
                # col_norm = mpl.colors.Normalize(0, c_range)

            n_data = len(nldat.inttimes)
            i = 0

            for x, y, ti, ref_val, label in zip(
                nldat.x, nldat.y, nldat.inttimes, nldat.ref_val, nldat.label
            ):
                if not plot_data:
                    break
                if i % 1000 == 0:
                    print("plot_sigproc_nl() i: %d / %d" % (i, n_data))
                i += 1

                if use_it_colors not in [0, None, False]:
                    # self.dbg_ti = ti
                    # c_idx = ti.astype(int)
                    # clip begrenzt auf 0..1
                    c_idx = (
                        (ti / use_it_colors).clip(0, 1) * (len(ti_colors) - 1)
                    ).astype(int)
                    # if c_idx[-1] > len(ti_colors) - 1:
                    #    print("C_idx", c_idx, "LEN", len(ti_colors))
                    data_color = ti_colors[c_idx]

                    # self.dbg_data_colors.append(data_color)
                    # self.dbg_c_idx.append(c_idx)
                    # reffun_color = data_color[-1]
                    # data_color = mpl.cm.jet(col_norm(ti))
                    # reffun_color = data_color[-1] # Farbe mit höchster Inttime nehmen

                if plot_data:
                    y_scaled = y
                    x_ = x
                    if type(plot_data) in [int]:
                        y_scaled = y[::plot_data]
                        x_ = x[::plot_data]
                    # print("plotte data")
                    if it_corr is not None:
                        y_scaled = y
                    if use_it_colors:
                        ax.scatter(
                            x_,
                            y_scaled,
                            c=data_color,
                            s=1,
                            edgecolors=None,
                            alpha=1,
                            zorder=2,
                            picker=use_picker,
                            pickradius=pickradius,
                            label=label,
                        )
                    else:
                        # AAAAAAAAAAAAARGH, use_ti_colors does not work with line plots!
                        # lineplot still kept, implement Infohandler for different Artists
                        print(plot_data_param)
                        ax.plot(
                            x_,
                            y_scaled,
                            #".-",  # fmt moved to plot_data_param
                            c=data_color,
                            zorder=1,
                            picker=use_picker,
                            pickradius=pickradius,
                            label=label,
                            **plot_data_param
                        )

            # end for

            if plot_nl and nldat.nl_fun is not None:  # FIXME: Block überarbeiten
                x_min = np.min([x.min() for x in nldat.x])
                x_max = np.max([x.max() for x in nldat.x])
                X_a = np.linspace(x_min, x_max, num=200, endpoint=True)

                ax.plot(
                    X_a,
                    nldat.nl_fun(X_a),
                    # "-",
                    # lw=2,
                    c=nl_color,
                    zorder=5,
                    alpha=1,
                    label=nl_label,
                    **plot_nl_param,
                )

            if len(nldat.outlier_limits[0]) > 0:
                plt.plot(
                    nldat.outlier_limits[0],
                    nldat.outlier_limits[1],
                    c="magenta",
                )
                plt.plot(
                    nldat.outlier_limits[0],
                    nldat.outlier_limits[2],
                    c="magenta",
                )

            if nldat.use_raw_counts:
                ax.set_xlabel("$y$ / counts")
            else:
                ax.set_xlabel("($y-y_{\mathrm{0}})$ / DN")
            ax.set_ylabel(
                # "$(y-y_{\mathrm{0}})/(t_\mathrm{i} \cdot y_\mathrm{ref})$"
                "normalized count rate"
            )
            # print("ACHTUNG: y_ref ist falscher Ausdruck, da das nicht y_ref ist, "
            #      "sondern (y/ti)_ref, also eine Referenzrate!")
            ax.set_title(title)
            # ax.set_xlim(0, 4095)  # FIXME
            # ax.set_ylim(0.89, 1.06)
            ax.grid(True)
            # fig.legend()
            fig.set_tight_layout(True)

            if filename:
                self.save_figure(fig, filename)

            if use_picker:
                InfoPicker(infodict=nldat.info).connect(fig)

            # plt.show()

        finally:
            # plt.ion()
            pass

        return fig

    def plot_sigproc_corrected(self, dat_lincorr):
        """
        Plot normalized count rates with corrected sigproc-nl.

        Ist eigentlich wie calc_sigproc_nl, aber auf korrigierten Daten.
        Kann man die Funktion einfach nochmal aufrufen? Das finden des
        Referenzwertes wäre evtl. anders. evtl. einfach Mittelwert?
        Aber es bleibt ja auch evtl. eine Rest-NL

        Takes corrected NLSelectedData as input and calculates nl like calc_nl

        """
        pass

    def plot_diode_nl(
        self,
        param,
        nlcal,
        sd=None,
        fig=None,
        datacolor=None,
        nlcolor=None,
        ticolors=None,
    ):

        """
        FIXME: noch nicht angepasst

        sollte nicht auf Standardansicht plotten, sondern einfach beide
        NL-Kompensationen anwenden. Messpunkte sollten dann gleichmäßig um
        eine horizontale Gerade verteilt sein

        plottet Ergebnis des Fits mit rücktransformierten Messwerten.
        Anstelle von direkten Countwerten werden wieder Countraten
        berechnet bzw die Fitergebnisse verwendet und auf Wert bei
        Referenzcounts skaliert. Damit kann man die Punkte unter die
        NL-CSpline plotten
        """

        if fig is None:
            fig = plt.figure("Pixel-Diode Nonlinearity", fig=self.figsize)
        if type(fig) == str:
            fig = plt.figure(fig)
        # if sd is None:
        #    sd = self._supportdata
        ax = fig.gca()

        if ticolors not in [0, None, False]:
            c_range = 100  # int(use_it_colors) + 1 # Anzahl der zu verwendenden Farben
            ti_colors = plt.cm.jet(np.linspace(0.1, 0.9, c_range))[:, 0:3]

        # if ticolors is not None:
        #     c_range = int(ticolors) + 1
        #     ti_colors = plt.cm.jet(np.linspace(0.1,0.9, c_range))[:,0:3]

        """
        Bei ZIM wurden Indizies gespeichert.
        Muss man hier die Residuenfunktion nachbauen?

        supportdata gibt es auch nicht. Hier müsste man das dat_obs-Objekt
        verwenden., bzw den kompletten args-Parameter aus dem Fit

        Funktion baut hier im wesentlichen die Residuumsfunktion
        nach, um Modellwerte zu berechnen und diese dann
        geeignet zu normieren.


        """
        yAD, A, regdata = nlcal.static_data
        with_diodenl = len(param.nldiodeparam) == 2
        print("WITH DIODENL:", with_diodenl)

        # res = np.array([], dtype=np.float64)
        # i = 0 # pointer auf Position in res
        i_B = 0  # pointer auf Sequenz
        for i_reg, rdat in enumerate(
            regdata.dat_bright
        ):  # rdat ist region-data -> enthält mehrere Sequenzen
            for (
                d
            ) in rdat:  # d ist sequenzdata einer inttime-serie einer Region

                # Hellsignal
                y_meas = d.mean  # gemessen
                y = y_meas - yAD[i_reg]  # yAD der jeweiligen Region abziehen

                corr_diodenl = 0
                if with_diodenl:
                    corr_diodenl = 0  # aus Parameterfeld berechnen

                y_krel = y / (
                    (A[i_reg] + param.B[i_B])
                    * (d.inttimes + param.timeoffset[0])
                )

                if ticolors not in [None, False]:
                    c_idx = (
                        (len(ti_colors) - 1) / ticolors * d.inttimes
                    ).astype(int)
                    # c_idx = d.inttimes.astype(int) # FIXME: funktioniert nur auf 1sek Auflösung
                    # FIXME: auf 0:c_range begrenzen
                    datacolor = ti_colors[c_idx]  # überschreiben

                lines = ax.scatter(
                    y, y_krel, c=datacolor, s=5
                )  # k_rel auf y_nl bezogen
                i_B = i_B + 1  # Sequenzzähler

        # NL-Kurve

        # nlcolor = None
        # if nlcolor is None:
        #    nlcolor = lines[0].get_color()  # Farbe vererben

        nlx = np.arange(4096)
        nly = nlcal.nl_fun(nlx, param.nlfuncparam)
        nlxc = nlcal._cspline_x(len(param.nlfuncparam))
        nlyc = nlcal.nl_fun(nlxc, param.nlfuncparam)
        ax.plot(nlx, nly, "-", label="$k_{rel}$", c="black", lw=2)
        ax.plot(nlxc, nlyc, "x", c="black")

        # merken zur Skalierung
        nl_min = nly.min()
        nl_max = nly.max()

        ax.set_ylim(nl_min * 0.98, nl_max * 1.02)
        ax.set_xlim(0, 4095)
        # ax.legend()
        ax.set_xlabel("$(y-y_{AD}) \qquad$ / $\qquad$ counts")
        ax.grid(True)
        # if self.ref_nl_with_yad:
        # else:
        #     ax.set_xlabel("$y \qquad$ / $\qquad$ counts")
        ax.set_ylabel("$(y-y_{AD})\quad/\quad((A+B)\cdot t_i)$\n$k_{rel}$")
        fig.set_tight_layout(True)
        ax.set_title("Nonlinearity")
        # save_figure(fig, "nonlinearity", cfg)
        return
