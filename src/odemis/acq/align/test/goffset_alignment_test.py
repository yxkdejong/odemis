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
from unittest.mock import patch


CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
SPARC_CONFIG = CONFIG_PATH + "sim/sparc2-focus-test.odm.yaml"

logging.getLogger().setLevel(logging.DEBUG)

class TestAutoAlignGratingDetectorOffsets(unittest.TestCase):
    """
    Test automatic grating-detector offset alignment
    """

    @classmethod
    def setUpClass(cls):
       # testing.start_backend(SPARC_CONFIG)

        cls.spgr = model.getComponent(role="spectrograph")
        cls.ccd = model.getComponent(role="ccd")
        cls.selector = model.getComponent(role="spec-det-selector")

    def setUp(self):
        # Speed up acquisition
        self.ccd.exposureTime.value = self.ccd.exposureTime.range[0]

    @timeout(100)
    def test_cancel(self):
        """
        Test cancelling auto-alignment
        """
        f = AutoAlignGratingDetectorOffsets(spectrograph=self.spgr, detectors=[self.ccd],)
        time.sleep(1)

        cancelled = f.cancel()
        self.assertTrue(cancelled)
        self.assertTrue(f.cancelled())

        with self.assertRaises(CancelledError):
            f.result(timeout=900)


    @timeout(1000)
    def test_multi_detector_iteration(self):
        """
        Test alignment iteration logic with multiple detectors
        """
        #spccd = model.getComponent(role="sp-ccd")
        #spccd.exposureTime.value = spccd.exposureTime.range[0]
        # f = AutoAlignGratingDetectorOffsets(spectrograph=self.spgr, detectors=[self.ccd, spccd], selector=self.selector)

        f = AutoAlignGratingDetectorOffsets(spectrograph=self.spgr, detectors=[self.ccd], selector=self.selector)
        res = f.result(timeout=900)

        n_gratings = len(self.spgr.axes["grating"].choices)
       # n_detectors = len(detectors)  # or 1 if you pass only self.ccd
        n_detectors = 1
        
        expected = n_detectors + (n_gratings - 1)

        self.assertEqual(len(res), expected)

        first_grating = list(self.spgr.axes["grating"].choices.keys())[0]
        dets_first = [d for (g, d) in res.keys() if g == first_grating]

        self.assertEqual(len(dets_first), n_detectors)

    @timeout(100)
    def test_cancel(self):
        f = AutoAlignGratingDetectorOffsets(spectrograph=self.spgr, detectors=[self.ccd],)

        time.sleep(1)

        cancelled = f.cancel()
        self.assertTrue(cancelled)
        self.assertTrue(f.cancelled())

        with self.assertRaises(CancelledError):
            f.result(timeout=900)

if __name__ == '__main__':
    unittest.main()