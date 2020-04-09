import local_packages

import numpy as np
import numpy.random as npr
import scipy.linalg as spl

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


class SimFlatsat(Sim):
    """Simulates a flatsat sitting on a table in a lab. Includes earth
    rate."""

    def __init__(self,
                 lon           = -122.41 * np.pi/180.0,
                 lat           = 37.7749 * np.pi/180.0,
                 T_enu_to_body = np.identity(3)):
        """Constructor for the flatsat simulation.

        Args:
            lon            longitude of flatsat (rad)
            lat            planetodetic/geodetic latitude of flatsat (rad)
            T_enu_to_body  attitude matrix of body with respect to local
                           ENU frame
        """

        self.lon              = lon
        self.lat              = lat
        self.t                = 0.0
        self.w_pcpf_wrt_inrtl = np.array([ 0.0,
                                           0.0,
                                           2.0 * np.pi / (23 * 3600.0 + 56 * 60.0 + 4.091)])
        self.w_body_wrt_pcpf  = np.zeros(3)
        self.r_pcpf           = planetodetic_to_pcpf(lon, lat)
        self.T_inrtl_to_pcpf  = np.identity(3)
        self.T_pcpf_to_enu = compute_T_pcpf_to_enu(self.r_pcpf)
        self.T_enu_to_body    = T_enu_to_body
        self.r_inrtl          = self.T_inrtl_to_pcpf.T.dot(self.r_pcpf)
        self.initialize_gyro_bias()
        self.initialize_log() # do last

    @property
    def T_inrtl_to_body(self):
        return self.T_enu_to_body.dot(self.T_pcpf_to_enu.dot(self.T_inrtl_to_pcpf))
                
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
        self.t += dt
        last_update = self.t

        # Create qEKF
        kf = Qekf(dt, pq.from_matrix(self.T_inrtl_to_body))
        
        while self.t < start + duration:
            self.T_inrtl_to_pcpf = compute_T_inrtl_to_pcpf(self.t - start,
                                                           self.w_pcpf_wrt_inrtl[2])
            self.r_inrtl = self.T_inrtl_to_pcpf.T.dot(self.r_pcpf)
            self.T_pcpf_to_enu = compute_T_pcpf_to_enu(self.r_pcpf)

            w_body_wrt_inrtl_inrtl = self.w_pcpf_wrt_inrtl + self.w_body_wrt_pcpf

            # Don't use += here because we want to create a new array, not copy it.
            self.b_gyro_body = self.b_gyro_body + npr.randn(3) * self.sigma_gyro_arw
            
            T_inrtl_to_body        = self.T_inrtl_to_body
            w_body_wrt_inrtl_body  = T_inrtl_to_body.dot(w_body_wrt_inrtl_inrtl)
            # The above is the gyroscope measurement.

            w_meas = w_body_wrt_inrtl_body + self.b_gyro_body

            kf.propagate(w_meas)
            if update_every is not None and self.t >= last_update + update_every:
                last_update = self.t

                update_args = self.make_vector_attitude_measurements()
                kf.update_attitude(*update_args)
                

            self.log['t'].append(self.t)
            self.log['qIB'].append(pq.Quat.from_matrix(T_inrtl_to_body))
            #self.log['rI'].append(self.r_inrtl)
            self.log['wBI'].append(w_body_wrt_inrtl_body)
            self.log['bg'].append(self.b_gyro_body)
            

            self.t += dt
        
        
        # Finish up by stacking logs

        self.log['t'] = np.hstack(self.log['t'])
        #self.log['rI'] = np.vstack(self.log['rI']).T
        self.log['wBI'] = np.vstack(self.log['wBI']).T
        self.log['bg']  = np.vstack(self.log['bg']).T

        return kf



        
if __name__ == '__main__':

    loader = SpiceLoader()

    sim = SimFlatsat()
    kf = sim.run(240.0, update_every = 1.0)
    kf.finish()

    plot_attitude_errors(sim, kf)
    plot_angular_velocity_measurement_check(sim, kf)
    plot_gyroscope_bias_errors(sim, kf)
    plot_gyroscope_bias_check(sim, kf)
    
    plt.show()
    
    
