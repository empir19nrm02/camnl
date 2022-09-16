# -*- coding: utf-8 -*-
"""
Interactive picking, highlighting and info display for matplotlib plots.

Clicking on lines or scatter plots hovers the clicked datasets and displays
stored info about them. When plotting many lines or scatter sets in one plot
(where it is not possible to use label) this is useful to find out which lines
have a distictive behaviour.

Selected lines can be reselectet (subset selection), new lines added or all
selections deleted.

The selected lines get hovered and the linecolor / marker size is changed.

@author: Christian Schrader
@email: christian.schrader@ptb.de
"""


from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib as mpl


# class LineInfoFormatter(object):

#     def format()

ModKeys = namedtuple(
    "ModKeys", ["SHIFT", "ALT", "CTRL"], defaults=[False, False, False]
)


class ModifierKeys:
    CTRL = False
    SHIFT = False
    ALT = False


"""
achtung, man könnte auch line.properties() oder pc.properties() verwenden,
um alles auszulesen. Scheint aber langsam zu sein:
%timeit line.properties()
1 ms ± 6.83 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit pc.properties()
987 µs ± 3.11 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Es fehlt die Möglichkeit, eigene Property-Handler zu registrieren
Man könnte auch in die Klasse hineinschreiben, für welche Artists sie
verantworlich sein können
"""


class _InfoPicker_BaseProperties(object):
    """Basis-Class / Template"""

    _artists = []  # [mpl.lines.Line2D]

    def backup(self, artist):
        pass

    def restore(self, artist):
        pass

    def highlight(self, artist):
        pass


class _InfoPicker_LineProperties(object):
    zorder = 0
    color = None
    linestyle = None
    linewith = None
    alpha = None

    def backup(self, line):
        self.color = line.get_color()
        self.zorder = line.get_zorder()
        self.linestyle = line.get_linestyle()
        self.linewidth = line.get_linewidth()
        # print("backup line", line, self.color, self.zorder, self.linestyle, self.linewidth)
        return

    def restore(self, line):
        # wenn man hier hl_props mit angibt, dann bräuchte man nur die Werte speichern, die später auch gesetzt werden, entsprechend in backup()
        # spart das Zeit?
        # print("restore line", line, self.color, self.zorder, self.linestyle, self.linewidth)
        line.set_color(self.color)
        line.set_zorder(self.zorder)
        line.set_linestyle(self.linestyle)
        line.set_linewidth(self.linewidth)
        return

    def highlight(self, line, hl_props={}):  # ist ein dict sinnvoll?
        # alles etwas umständlich, weil Werte in hl_props nicht gesetzt sein müssen
        color = hl_props.get("color")
        zorder = hl_props.get("zorder")
        linestyle = hl_props.get("linestyle")
        linewidth = hl_props.get("linewidth")
        if color:
            line.set_color(color)
        if zorder:
            line.set_zorder(zorder)
        if linestyle:
            line.set_linestyle(linestyle)
        if linewidth:
            line.set_linewidth(linewidth)
        # print("highlight line", line, color, zorder, linestyle, linewidth)
        return


class _InfoPicker_PathCollectionProperties(object):
    zorder = 0
    edgecolors = None
    facecolors = None
    marker = None
    # linestyle = None
    # linewith = None
    sizes = None

    def backup(self, pc):
        self.zorder = pc.get_zorder()
        self.edgecolors = pc.get_edgecolors()
        self.facecolors = pc.get_facecolors()
        self.sizes = pc.get_sizes()
        # print("backup pathcollection", pc, self.zorder, self.sizes)
        return

    def restore(self, pc):
        pc.set_zorder(self.zorder)
        pc.set_edgecolors(self.edgecolors)
        pc.set_sizes(self.sizes)
        # pc.set_facecolors(self.facecolors)
        # print("restore pathcollection", pc, self.zorder, self.sizes)
        return

    def highlight(self, pc, hl_props={}):  # ist ein dict sinnvoll?
        # alles etwas umständlich, weil Werte in hl_props nicht gesetzt sein müssen
        edgecolors = hl_props.get("colors")
        # facecolors = hl_props.get('colors')
        zorder = hl_props.get("zorder")
        sizes = hl_props.get("sizes")
        if edgecolors:
            pc.set_edgecolors(edgecolors)
        if zorder:
            pc.set_zorder(zorder)
        if sizes:
            pc.set_sizes(sizes)
        # print("highlight pathcollection", pc, zorder, sizes) # edgecolors zu gross zum ausgeben
        return


"""
Da eh nur Artists verwendet werden können, für die eine Klasse implementiert
wurde, kann man auch den Cache hartverdrahten und muss nicht mit dicts arbeiten
->  {artist.__class__ : propdata}
"""
# class _InfoPicker_PropertyCache(object):
#    PathCollection = None
#    Line2D = None


class InfoPicker(object):
    _modifier = ["shift", "control", "alt", "ctrl"]
    _property_cache = {}
    _picked_artists = set()

    def __init__(
        self,
        hl_color="magenta",
        hl_zorder=10,
        infodict=None,
        label_prefix="_seq",
    ):

        self._picked_artists = set()
        self._property_cache = {}

        # self.active_lines = []
        # self.orig_color = []
        # self.orig_zorder = []
        # self.colorcycle = plt.rcParams['axes.prop_cycle']()
        self.hl_color = hl_color
        self.hl_zorder = hl_zorder
        self.infodict = infodict
        self.label_prefix = label_prefix
        # self.active_modifier = ModifierKeys()
        self.active_modifier = ModKeys()
        return

    def connect(self, fig):
        self.callback_id_pick = fig.canvas.mpl_connect(
            "pick_event", self.onpick_handler
        )
        self.callback_id_buttonPress = fig.canvas.mpl_connect(
            "button_press_event", self.button_press_handler
        )
        self.callback_id_buttonRelease = fig.canvas.mpl_connect(
            "button_release_event", self.button_release_handler
        )
        self.callback_id_keyPress = fig.canvas.mpl_connect(
            "key_press_event", self.key_press_handler
        )
        self.callback_id_keyRelease = fig.canvas.mpl_connect(
            "key_release_event", self.key_release_handler
        )
        fig._pick_handler = self
        self.figure = fig
        return

    def button_press_handler(self, event):
        """pick dispatcher"""
        # print("klick")
        if event.button == mpl.backend_bases.MouseButton.LEFT:
            self.action_mb_left(event)
            return
        elif event.button == mpl.backend_bases.MouseButton.RIGHT:
            self.action_mb_right(event)
            return
        return

    def button_release_handler(self, event):
        # print("release")
        self._picked_artists = set()  # alle gepicken Elemente wieder vergessen
        self.figure.canvas.draw()  # neuzeichnen
        return

    def action_mb_left(self, event):
        """what to do on left mouse click

        - wenn keine gepickten Objekte da sind, nix tun
        - wenn gepickte Objekte da sind, diese highlighten,  alle anderen löschen
          -> zum einfachen durchklicken
          - bei CTRL wieder löschen (togglen) geht nicht sinnvoll, da man nicht
            ausschließlich einzelne Objekte anklicken kann. Man würde in dem
            Fall einige hinzufügen (die neuen) und einige löschen (die bereits
            gehighligteten)
        - wenn CTRL und gepickte Objekte da sind, diese zu gehighlighteten hinzufügen -> so ist man das irgendwie gewohnt
        - wenn SHIFT+CRTL und gepickte Objekte da sind,
        """

        if not self._picked_artists:
            return  # nix zu tun

        """
        erst bestimmen, welche Objekte zurückgesetzt werden müssen und
        welche aktiviert werden müssen
        """

        # ggfs ALT hinzufügen
        if not self.active_modifier.CTRL and not self.active_modifier.SHIFT:
            # aktuelle Auswahl restoren, gepickte highlighten -> ersetzen

            already_highlighted = set(self._property_cache.keys())

            # die Artists, von den gehighlighteten, die nicht in den gepickten sind
            artists_to_restore = already_highlighted.difference(
                self._picked_artists
            )

            # artists von den gepickten, die noch nicht in den gehighlighteten sind
            artists_to_highlight = self._picked_artists.difference(
                already_highlighted
            )

            # print("------------")
            # print(already_highlighted)
            # print(artists_to_restore)
            # print(artists_to_highlight)
            # print("------------")

        elif self.active_modifier.CTRL and not self.active_modifier.SHIFT:
            artists_to_highlight = (
                self._picked_artists
            )  # alle zu gehighlighteten hinzufügen
            artists_to_restore = set()  # keine resetten

        elif self.active_modifier.CTRL and self.active_modifier.SHIFT:
            # gepickte Objekte auf bereits aktive beschränken -> subselection
            # ist eigentlich eher ein selektives Löschen als ein Adden

            already_highlighted = set(self._property_cache.keys())

            # die Artists, von den gehighlighteten, die nicht in den gepickten sind
            artists_to_restore = already_highlighted.difference(
                self._picked_artists
            )

            # es werden keine neuen artists gehighlightet
            artists_to_highlight = set()
            # testen, ob gehighlightete Artists über bleiben würden
            # wenn ja, Operation anwenden, wenn nicht, nicht alle zurücksetzen
            if already_highlighted == artists_to_restore:
                artists_to_restore = set()
        else:
            return

        self.restore_artist_properties(artists_to_restore)

        for a in artists_to_highlight:
            self.backup_artist_properties(a)
            self.highlight_artist(a)

        for (
            a
        ) in (
            self._property_cache.keys()
        ):  # aktualisierte Version von already_highlighted
            # Info der aktuell gehighlighteten Artists ausgeben
            # muss man das auch irgendwie als extra Funktion machen?
            # eventuell auch an i-taste binden, infos über alle gehighlighteten Objekte
            self.print_info(a)
        if len(self._property_cache) > 0:
            print()  # info-block etwas absetzen

        return

    def action_mb_right(self, event):
        """what to do on right mouse click

        - wenn keine gepickten Objekte da sind, nix tun
        - gepickte Objekte resetten, wenn sie gehighlightet sind.
        - wenn CTRL und gepickte Objekte da sind, diese zu gehighlighteten hinzufügen -> so ist man das irgendwie gewohnt


        - wenn SHIFT gedrückt ist, alle Objekte resetten
        - oder umgekehrt?
        """

        # highlighted = set(self._property_cache.keys()) # nur fürs print

        artists_to_restore = set()

        if not self.active_modifier.CTRL and not self.active_modifier.SHIFT:
            if self._picked_artists:
                # nur
                highlighted = set(self._property_cache.keys())
                artists_to_restore = self._picked_artists.intersection(
                    highlighted
                )
            else:
                # nichts löschen, da man zu leicht neben eine Linie klickt
                return

        elif not self.active_modifier.CTRL and self.active_modifier.SHIFT:
            if not self._picked_artists:
                # alles restoren
                artists_to_restore = set(
                    self._property_cache.keys()
                )  # wichtig: nicht keys-objekt direkt nehmen! sonst ändert sich iterable während Objekte aus dict gelöscht werden
            else:
                pass

        elif self.active_modifier.CTRL and self.active_modifier.SHIFT:
            pass
        else:
            return

        self.restore_artist_properties(artists_to_restore)
        return

    def backup_artist_properties(self, a):
        props = self._property_cache.get(a, None)
        if props is not None:
            logger_picker.debug("Property already stored! {!s}".format(a))
            return
        else:

            # ----------------------------------------------------------------
            # FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
            # diese hartverdrahtete weiche sollte man durch eine Möglichkeit
            # ersetzen, eigene Klassen für einen Artist-Typ zu registrieren
            if isinstance(a, mpl.lines.Line2D):
                props = _InfoPicker_LineProperties()
            elif isinstance(a, mpl.collections.PathCollection):
                props = _InfoPicker_PathCollectionProperties()

            # ----------------------------------------------------------------

        props.backup(a)  # kann hier irgendwas schief gehen? try-except?
        self._property_cache[a] = props
        return

    def restore_artist_properties(self, artists):
        # print("ART", artists)
        for a in artists:
            props = self._property_cache.get(a, None)
            if props is None:
                logger_picker.debug("No Properties for artist! {!s}".format(a))
            props.restore(a)  # kann hier irgendwas schief gehen? try-except?
            self._property_cache.pop(a)  # man könnte oben schon pop nehmen
        return

    def highlight_artist(self, a):
        props = self._property_cache.get(a, None)
        if props is None:
            logger_picker.debug("No Properties for artist! {!s}".format(a))
        props.highlight(
            a,
            {"color": self.hl_color, "zorder": self.hl_zorder, "sizes": [20]},
        )  # FIXME: Parameterfestlegung und -übergabe
        return

    def onpick_handler(self, event):
        """
        sammelt Daten der gepickten Objekte, Typ ist hier egal

        pick events kommen vor button-press-events. Deshalb hier Objekte
        sammeln und bei button-down entscheiden, was damit geschieht.

        - wenn kein CTRL gedrückt ist, alle picken
        - wenn CTRL gedrückt ist, nur Objekte picken, die schon aktiviert sind
          -> verfeinerung
          Was damit geschieht, hängt von button ab.
        """
        # print("onpick")

        if self.figure.canvas.toolbar.mode is not mpl.backend_bases._Mode.NONE:
            logger_picker.debug("Wrong toolbar mode!")
            return
        # erstamal alle gepickten Elemente sammeln, da pick-event vor button-events kommt.
        self._picked_artists.add(event.artist)
        return

    # ------------------------ Key-Handling -----------------------------------

    def key_press_handler(self, event):
        # logger_picker.debug("press: {:s}".format(event.key))
        self.handle_modifier(event.key, True)
        return

    def key_release_handler(self, event):
        # logger_picker.debug("release: {:s}".format(event.key))
        self.handle_modifier(event.key, False)
        return

    def handle_modifier(self, key, active):
        active = bool(active)
        shift = alt = ctrl = False
        for k in key.split("+"):
            if k == "shift":
                shift = True
            if k == "alt":
                alt = True
            if k in ["control", "ctrl"]:
                ctrl = True

        mk = ModKeys(shift, alt, ctrl)
        if active:
            # m_eff = np.bitwise_or(self.active_modifier, mk)  # Bits hinzufügen
            m_eff = [a or b for a, b in zip(self.active_modifier, mk)]
        else:
            # m_eff = np.bitwise_and(self.active_modifier, np.bitwise_not(mk))  # Bits löschen
            m_eff = [
                a and not b for a, b in zip(self.active_modifier, mk)
            ]  # braucht kein numpy und ist etwas schneller
        self.active_modifier = ModKeys(*m_eff)

        # logger_picker.debug("MOD: SHIFT: {:b} ALT: {:b} CTRL {:b}".format(
        #     self.active_modifier.SHIFT,
        #     self.active_modifier.ALT,
        #     self.active_modifier.CTRL))

        # diese Ausgabe ist etwas kompakter, besser zu erfassen

        # logger_picker.debug("MOD: {:s}{:s}{:s}".format(
        #     "S" if self.active_modifier.SHIFT else "_",
        #     "A" if self.active_modifier.ALT else "_",
        #     "C" if self.active_modifier.CTRL else "_"))
        return

    # ------------------------------------------------------------------------

    def print_info(self, line):
        lab = line.get_label()
        # print("LineLabel:", lab)
        if lab.startswith(self.label_prefix):
            line_id = int(lab[len(self.label_prefix) :])
            # Objekt im Infodict muss String zurückggeben. Kann also auch
            # gleich ein extern formatierter String sein.
            print(
                "Info: ID: {:d}: {:s}".format(
                    line_id, str(self.infodict[line_id])
                )
            )
        return
