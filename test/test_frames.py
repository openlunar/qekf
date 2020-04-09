import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import frames

import numpy as np

class TestFrames(unittest.TestCase):

    def test_pcpf_to_planetodetic(self):
        r_pcpf = np.array([0.0, 0.0, 6378137.0])
        lon, lat, alt = frames.pcpf_to_planetodetic(r_pcpf)
        self.assertEqual(lat, np.pi/2.0)
        self.assertEqual(lon, 0.0)
        self.assertEqual(alt, 0.0)
