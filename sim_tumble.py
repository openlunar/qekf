import local_packages

import numpy as np
import numpy.random as npr
import scipy.linalg as spl
from scipy.integrate import DOP853

import qekf # import before pyquat
import pyquat as pq

from spice_loader import SpiceLoader

from frames import planetodetic_to_pcpf
from frames import compute_T_pcpf_to_enu
from frames import compute_T_inrtl_to_pcpf

from qekf import Qekf
from sim import Sim

from physics import lookup_u_sun_inrtl, lookup_u_mag_enu, noisify_line_of_sight_vector

from plots import plot_attitude_errors
from plots import plot_gyroscope_bias_errors
from plots import plot_angular_velocity_measurement_check
from plots import plot_gyroscope_bias_check
import matplotlib.pyplot as plt

class SimTumble(Sim):
    """Simulates an orbiting satellite."""

    def __init__(self, x_inrtl, J,
                 q_inrtl_to_body  = pq.identity(),
                 w_body_wrt_inrtl = np.zeros(3)):
        """Constructor for the tumbling satellite simulation.

        Args:
            r_inrtl           position and velocity in inertial 
                              frame (m, m/s)
            J                 moment of inertia matrix
            q_inrtl_to_body   pyquat.Quat attitude of the body frame
                              with respect to the inertial frame
                              (default is the identity quaternion)
            w_body_wrt_inrtl  initial angular velocity of the body with
                              respect to the inertial frame, expressed
                              in the inertial frame

        """
        self.x_inrtl          = x_inrtl
        self.J                = J
        self.Jinv             = spl.inv(J)
        self.t                = 0.0
        self.w_body_wrt_inrtl = w_body_wrt_inrtl
        self.q_inrtl_to_body  = q_inrtl_to_body
        self.T_inrtl_to_pcpf  = np.identity(3)
        self.T_pcpf_to_enu    = compute_T_pcpf_to_enu(self.T_inrtl_to_pcpf.dot(self.r_inrtl))
        self.initialize_gyro_bias()
        self.initialize_log()

    @property
    def T_inrtl_to_body(self):
        return self.q_inrtl_to_body.to_matrix()

    @property
    def T_enu_to_body(self):
        return self.T_inrtl_to_body.dot(self.T_inrtl_to_pcpf.T.dot(self.T_pcpf_to_enu.T))
    
    @property
    def r_inrtl(self):
        return self.x_inrtl[0:3]

    @property
    def v_inrtl(self):
        return self.x_inrtl[3:6]

    def step(self):
        # Step position/velocity
        self.integ.step()
        if integ.t < self.t:
            raise StandardError("integration didn't complete, refactor needed")
        self.x_inrtl = self.integ.y

        # Step attitude and angular velocity
        self.q_inrtl_to_body, self.w_body_wrt_inrtl = pq.step_cg3(self.q_inrtl_to_body, self.w_body_wrt_inrtl, J = self.J, Jinv = self.Jinv)        

    def run(self, duration,
            dt           = 0.1,
            update_every = None):
        """Run the simulation until the provided duration of time has elapsed,
        with or without updates.

        Args:
            duration      length of time to run for (s)
            dt            propagation time step (s; defaults to 0.1)
            update_every  frequency of unit vector measurement updates
                          (default is None, which turns them off)

        Returns:
            The qEKF that was created.

        """

        start = self.t

        # This is not the best way to create one of these, but we
        # don't really care about position and velocity all that much
        # right now. We only care about how they affect the magnetic
        # field vector.
        self.integ = DOP853(gravity, start, self.x_inrtl, start + duration,
                            first_step = dt,
                            max_step   = dt)

        # Create qEKF
        kf = Qekf(dt, pq.from_matrix(self.T_inrtl_to_body))
                       
        self.t += dt
        last_update = self.t
        self.step()

        while self.t < start + duration:
            self.T_inrtl_to_pcpf = compute_T_inrtl_to_pcpf(self.t - start,
                                                           self.w_pcpf_wrt_inrtl[2])

            self.T_pcpf_to_enu   = compute_T_pcpf_to_enu(self.T_inrtl_to_pcpf.dot(self.r_inrtl))
