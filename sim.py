import local_packages

import numpy as np
import numpy.random as npr
import scipy.linalg as spl

import qekf # import before pyquat
import pyquat as pq

from spice_loader import SpiceLoader

from frames import planetodetic_to_pcpf
from frames import compute_T_inrtl_to_pcpf

from qekf import Qekf

from physics import lookup_u_sun_inrtl, lookup_u_mag_enu, noisify_line_of_sight_vector

from plots import plot_attitude_errors
from plots import plot_gyroscope_bias_errors
from plots import plot_angular_velocity_measurement_check
from plots import plot_gyroscope_bias_check
import matplotlib.pyplot as plt

class Sim(object):
    # Made up these numbers, all in radians
    sigma_sun_sensor = 1e-3
    sigma_mag_sensor = 1e-3
    sigma_sun_model  = 1e-11
    sigma_mag_model  = 1e-3

    # These numbers come from IMU specifications:
    sigma_gyro_bias  = 0.1 * np.pi/180.0 # turn-on bias (r/s)
    sigma_gyro_arw   = 0.0013 * np.pi/180.0

    def initialize_gyro_bias(self):
        self.b_gyro_body = npr.randn(3) * self.sigma_gyro_bias

    def initialize_log(self):
        self.log = {'t':   [self.t],
                    'qIB': [pq.from_matrix(self.T_inrtl_to_body)],
                    'rI':  [self.r_inrtl],
                    'wBI': [np.zeros(3)],
                    'bg':  [self.b_gyro_body]}
        
    def make_vector_attitude_measurements(self):
        """Use SPICE and WMM2015 to get sun and magnetic field
        vectors. Pretend these are "true."  Then corrupt those vectors
        with some noise, twice --- once to make reference vectors and
        another time to make observation vectors.

        """
        
        # Get reference and observation vectors
        u_sun_inrtl = lookup_u_sun_inrtl(self.t)
        u_sun_ref   = noisify_line_of_sight_vector(u_sun_inrtl, self.sigma_sun_model)
        u_sun_body  = self.T_inrtl_to_body.dot(u_sun_inrtl)
        u_sun_obs   = noisify_line_of_sight_vector(u_sun_body, self.sigma_sun_sensor)

        # Pretend WMM2015's value is "true" and not just a
        # model. Corrupt "true" to our "model", which is ref.
        # Then corrupt "true" to our observation.
        T_enu_to_inrtl  = self.T_inrtl_to_pcpf.T.dot(self.T_pcpf_to_enu.T)
        u_mag_enu       = lookup_u_mag_enu(self.lon, self.lat)
        u_mag_inrtl     = T_enu_to_inrtl.dot(u_mag_enu)
        u_mag_ref       = noisify_line_of_sight_vector(u_mag_inrtl, self.sigma_mag_model)
        u_mag_body      = self.T_inrtl_to_body.dot(u_mag_inrtl)
        u_mag_obs       = noisify_line_of_sight_vector(u_mag_body, self.sigma_mag_sensor)

        return (u_sun_obs, u_mag_obs, u_sun_ref, u_mag_ref)
