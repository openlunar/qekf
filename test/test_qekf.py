import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qekf import Qekf

import pyquat as pq
import numpy as np
import numpy.random as npr
import scipy.linalg as spl

class TestQekf(unittest.TestCase):

    def setUp(self):
        from spice_loader import SpiceLoader
        
        self.loader = SpiceLoader()
    
    def test_ongoing_gyro_bias_estimation(self):
        """The initial estimate of the gyro bias should be nonzero"""
        from sim_flatsat import SimFlatsat
        
        sim    = SimFlatsat()
        kf     = sim.run(60.0, update_every = 1.0)
        kf.finish()

        diff = kf.log['bg'][:,1] - kf.log['bg'][:,-1]        
        for ii in range(0,3):
            self.assertNotEqual(diff[ii], 0.0)

    
