import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim_flatsat import SimFlatsat

import scipy.linalg as spl


class TestSimFlatsat(unittest.TestCase):

    def test_planet_rotation(self):
        """Make sure planet is rotating the correct direction"""
        # Start flatsat at r_inrtl = [1,0,0]
        sim = SimFlatsat(lat = 0.0, lon = 0.0)
        self.assertEqual(sim.r_inrtl[1], 0.0)
        kf = sim.run(60.0) # run for a minute

        # Confirm that flatsat now has a negative y component
        self.assertLess(sim.r_inrtl[0] / spl.norm(sim.r_inrtl), 1.0)
        self.assertGreater(sim.r_inrtl[1], 0.0)
        self.assertEqual(sim.r_inrtl[2], 0.0)
        
