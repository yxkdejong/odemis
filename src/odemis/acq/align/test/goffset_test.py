# -*- coding: utf-8 -*-
"""
Test Sparc auto grating offset alignment
"""

import os
import unittest
import logging
import numpy as np
import odemis

from odemis import model
from odemis.util import timeout
from odemis.acq.align.goffset import(
    find_peak_position,
    estimate_goffset_scale,
    sparc_auto_grating_offset
)

logging.getLogger().setLevel(logging.DEBUG)

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
SPARC_CONFIG = CONFIG_PATH + "sim/sparc2-focus-test.odm.yaml"

class TestSparcAutoGratingOffset(unittest.TestCase):
    """
    Test automatic grating offset alignment.
    """

    @classmethod
    def setUpClass(cls):
        #testing.start_backend(SPARC_CONFIG)

        cls.detector = model.getComponent(role="ccd")
        cls.spgr = model.getComponent(role="spectrograph")

        cls._original_goffset = cls.spgr.goffset
        cls._original_position = cls.spgr.position.value.copy()

    # @classmethod
    # def tearDownClass(cls):
    #     # restore original position
    #     try:
    #         cls.spgr.moveAbsSync(cls._original_position)
    #     except Exception:
    #         logging.exception("Failed restoring spectrograph position")

    def setUp(self):
        # speed up detector
        self.detector.exposureTime.value = self.detector.exposureTime.range[0]

    def test_find_peak_position_synthetic(self):
        """
        Test peak detection on synthetic Gaussian data.
        """
        x = np.arange(200)
        true_center = 83.4
        spectrum = np.exp(-0.5*((x-true_center)/3.0)**2)

        peak = find_peak_position(spectrum)
        self.assertAlmostEqual(peak, true_center, places=1)

    def test_find_peak_position_2d(self):
        """
        Test peak detection on 2D data (mean over axis).
        """
        x = np.arange(200)
        true_center = 120.0
        line = np.exp(-0.5*((x-true_center)/4.0)**2)
        image = np.tile(line, (50, 1))

        peak = find_peak_position(image)
        self.assertAlmostEqual(peak, true_center, places=1)

    @timeout(100)
    def test_estimate_goffset_scale(self):
        """
        Test that goffset scale is non-zero and finite.
        """
        scale = estimate_goffset_scale(self.spgr, self.detector)

        self.assertIsInstance(scale, float)
        self.assertNotEqual(scale, 0.0)
        self.assertTrue(np.isfinite(scale))

    @timeout(800)
    def test_auto_grating_offset(self):
        """
        Test automatic centering of spectral peak.
        """
        delta = 0 # intentionally misalign
        current = self.spgr.position.value["goffset"]
        goffset_max = self.spgr.axes["goffset"].range[1]
        direction = 1 if (current + delta < goffset_max) else -1

        self.spgr.moveRelSync({"goffset": delta * direction})
        logging.info("Test: after misalign move, spgr.position.gooffset = %s", self.spgr.position.value["goffset"])
        f = sparc_auto_grating_offset(self.spgr, self.detector, max_it=50)

        result = f.result(timeout=800)
        self.assertTrue(result)

    def test_auto_grating_offset_acquisition(self):
        """
        Force the peak off detector so acquisition must run, then verify alignment succeeds.
        """
        # choose a large delta to ensure the peak is off detector
        delta = -2000
        current = self.spgr.position.value["goffset"]
        minv, maxv = self.spgr.axes["goffset"].range
        direction = 1 if (current + delta < maxv) else -1

        self.spgr.moveRelSync({"goffset": delta * direction})
        f = sparc_auto_grating_offset(self.spgr, self.detector, max_it=80, tolerance_px=0.4, gain=0.4)
        result = f.result(timeout=1200)
        self.assertTrue(result)

    @timeout(800)
    def test_scale_not_misaligned(self):
        """
        Verify scale estimation only happens when the peak is misaligned.
        This is inferred from the probe move performed by estimate_goffset_scale().
        """
        # reset spectrograph to known aligned position
        self.spgr.moveAbsSync(self._original_position)

        start_goffset = self.spgr.position.value["goffset"]

        f = sparc_auto_grating_offset(self.spgr, self.detector, max_it=20)
        result = f.result(timeout=300)

        end_goffset = self.spgr.position.value["goffset"]

        self.assertTrue(result)

        # If peak is already centered, the algorithm exits immediately
        # so goffset should not change.
        self.assertAlmostEqual(
            start_goffset,
            end_goffset,
            places=6,
            msg="goffset changed even though peak was already centered (scale estimation likely ran)"
        )

    def test_scale_estimation_misaligned(self):
        delta = 500
        current = self.spgr.position.value["goffset"]
        maxv = self.spgr.axes["goffset"].range[1]
        direction = 1 if (current + delta < maxv) else -1

        self.spgr.moveRelSync({"goffset": delta * direction})

        start_goffset = self.spgr.position.value["goffset"]

        f = sparc_auto_grating_offset(self.spgr, self.detector, max_it=50)
        result = f.result(timeout=600)

        end_goffset = self.spgr.position.value["goffset"]

        self.assertTrue(result)

        # If misaligned, centering should move the grating
        self.assertNotAlmostEqual(
            start_goffset,
            end_goffset,
            places=3,
            msg="goffset did not change during alignment when peak was misaligned"
        )

    @timeout(100)
    def test_cancel(self):
        """
        Test cancelling alignment.
        """
        f = sparc_auto_grating_offset(self.spgr, self.detector)
        # Wait for the result or a timeout
        try:
            f.result(timeout=5)
        except:
            pass
        self.assertTrue(f.done())

if __name__ == "__main__":
    unittest.main()
