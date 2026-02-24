import logging
import os
import time
import unittest
import numpy
from collections.abc import Iterable
from concurrent.futures import CancelledError
from scipy import ndimage

from odemis import model, acq
import odemis
from odemis.acq import align, stream, path
from odemis.acq.align.autofocus import Sparc2AutoFocus, MTD_BINARY
from odemis.acq.align.goffset_alignment import AutoAlignGratingDetectorOffsets
from odemis.dataio import tiff, hdf5
from odemis.util import testing, timeout, img
import odemis.util.focus

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
SPARC_CONFIG = CONFIG_PATH + "sim/sparc2-focus-test.odm.yaml"

class TestAutoAlignmentGratingDetectorOffsets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use SPARC config (same used in spectrometer autofocus tests)
        testing.start_backend(SPARC_CONFIG)

        cls.spectrograph = model.getComponent(role="spectrograph")
        cls.ccd = model.getComponent(role="ccd")
        cls.spccd = model.getComponent(role="sp-ccd")
        cls.selector = model.getComponent(role="spec-det-selector")

        cls.gratings = cls.spectrograph.axes["grating"].choices