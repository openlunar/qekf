import local_packages

import numpy as np
import numpy.random as npr
import scipy.linalg as spl

import qekf # import before pyquat
import pyquat as pq

from frames import planetodetic_to_pcpf
from frames import compute_T_pcpf_to_enu
from frames import compute_T_inrtl_to_pcpf

from qekf import Qekf

from spice_loader import SpiceLoader
from spiceypy import spiceypy as spice

import wmm2015 # see local_packages

def lookup_u_sun_inrtl(t):
    """Return the unit vector to the sun in an earth-centered inertial frame.

    Uses SPICE and treats t as an ephemeris time."""
    r = spice.spkezp(10, t, 'J2000', 'LT', 399)[0]
    return r / spl.norm(r)


def lookup_u_mag_enu(lat, lon, alt = 0.0, year = 2020):
    """Return the unit vector of the magnetic field in a local ENU frame."""

    # WMM takes degrees and km
    mag = wmm2015.wmm(lat * 180/np.pi, lon * 180/np.pi, alt / 1000.0, year)

    # Get vector in north/east/down coordinates.
    enu = np.array([ mag.east.item(),
                     mag.north.item(),
                    -mag.down.item()  ])
    return enu / spl.norm(enu)


def noisify_vector(u, sigma):
    # First rotate the u vector to be along z
    z = np.array([0.0, 0.0, 1.0])

    # get axis of rotation
    v = np.cross(u, z) 
    v /= spl.norm(v)

    # Get angle about that axis
    theta = np.arccos(u.dot(z))
    T_z_to_u = pq.Quat.from_angle_axis(theta, *v).to_matrix()
            
    # Now produce a misalignment matrix
    e = npr.rand(3) * sigma
    T_misalign = np.identity(3) - pq.skew(e)

    u_noisy = T_z_to_u.dot(T_misalign.dot(z))
    return u_noisy / spl.norm(u_noisy) # normalize it
    

class SimFlatsat(object):
    # Made up these numbers, all in radians
    sigma_sun_sensor = 1e-3
    sigma_mag_sensor = 1e-3
    sigma_sun_model  = 1e-11
    sigma_mag_model  = 1e-3
    sigma_gyro_bias  = 0.1 * np.pi/180.0 # turn-on bias (r/s)
    sigma_gyro_arw   = 0.0013 * np.pi/180.0
    
    def __init__(self,
                 lat = 37.7749 * np.pi/180.0,
                 lon = -122.41 * np.pi/180.0,
                 T_enu_to_body = np.identity(3)):

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
        self.b_gyro_body      = npr.randn(3) * self.sigma_gyro_bias

        self.log = {'t':   [self.t],
                    'qIB': [pq.from_matrix(self.T_inrtl_to_body)],
                    'rI':  [self.r_inrtl],
                    'wBI': [np.zeros(3)],
                    'bg':  [self.b_gyro_body]}

    @property
    def T_inrtl_to_body(self):
        return self.T_enu_to_body.dot(self.T_pcpf_to_enu.dot(self.T_inrtl_to_pcpf))
        

    def make_vector_attitude_measurements(self):
        """Use SPICE and WMM2015 to get sun and magnetic field
        vectors. Pretend these are "true."  Then corrupt those vectors
        with some noise, twice --- once to make reference vectors and
        another time to make observation vectors.

        """
        
        # Get reference and observation vectors
        u_sun_inrtl = lookup_u_sun_inrtl(self.t)
        u_sun_ref   = noisify_vector(u_sun_inrtl, self.sigma_sun_model)
        u_sun_body  = self.T_inrtl_to_body.dot(u_sun_inrtl)
        u_sun_obs   = noisify_vector(u_sun_body, self.sigma_sun_sensor)

        # Pretend WMM2015's value is "true" and not just a
        # model. Corrupt "true" to our "model", which is ref.
        # Then corrupt "true" to our observation.
        T_enu_to_inrtl  = self.T_inrtl_to_pcpf.T.dot(self.T_pcpf_to_enu.T)
        u_mag_enu       = lookup_u_mag_enu(self.lon, self.lat)
        u_mag_inrtl     = T_enu_to_inrtl.dot(u_mag_enu)
        u_mag_ref       = noisify_vector(u_mag_inrtl, self.sigma_mag_model)
        u_mag_body      = self.T_inrtl_to_body.dot(u_mag_inrtl)
        u_mag_obs       = noisify_vector(u_mag_body, self.sigma_mag_sensor)

        return (u_sun_obs, u_mag_obs, u_sun_ref, u_mag_ref)
                
    def run(self, duration, dt = 0.1, update_every = None):
        """Run the simulation, with or without updates."""
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


def dq_array(q_inrtl_to_bodyhat_list, q_inrtl_to_body_list):
    """Given a list of quaternions from the qEKF and a list from the sim
    which are aligned in time, return the error attitude (body to
    estimated body).

    Args:
        q_inrtl_to_bodyhat_list  list of quaternions from the filter
        q_inrtl_to_body_list     list of quaternions from the sim

    Returns:
        A 3xN matrix consisting of angle*axis attitude errors.
    """

    ary = []
    
    for ii,qIBh in enumerate(q_inrtl_to_bodyhat_list):
        qIB = q_inrtl_to_body_list[ii]

        ary.append( (qIBh * qIB.conjugated()).to_rotation_vector().reshape(3) )

    return np.vstack(ary).T
        
if __name__ == '__main__':

    loader = SpiceLoader()

    sim = SimFlatsat()
    kf = sim.run(240.0, update_every = 1.0)
    kf.finish()

    import matplotlib.pyplot as plt

    fig1, axes1 = plt.subplots(3, 1, sharex=True)
    fig2, axes2 = plt.subplots(3, 1, sharex=True)
    fig3, axes3 = plt.subplots(3, 1, sharex=True)
    fig4, axes4 = plt.subplots(3, 1, sharex=True)

    dq = dq_array(kf.log['qIB'], sim.log['qIB'])
    
    for ii in range(0, 3):
        axes1[ii].plot(sim.log['t'], dq[ii,:] * 180/np.pi, label='attitude error', alpha=0.6)
        axes1[ii].plot(sim.log['t'],  kf.log['sigma'][ii,:] * 180/np.pi, alpha=0.6, label='1-sigma', c='k')
        axes1[ii].plot(sim.log['t'], -kf.log['sigma'][ii,:] * 180/np.pi, alpha=0.6, c='k')
        
        axes1[ii].grid(True)
        
        axes2[ii].plot(sim.log['t'], sim.log['wBI'][ii,:] - kf.log['wm'][ii,:], label='delta', alpha=0.6)
        #axes[ii].plot(kf.log['t'],  kf.log['wm'][ii,:],  label='qekf', alpha=0.6)
        axes2[ii].grid(True)

        
        axes3[ii].plot(sim.log['t'], sim.log['bg'][ii,:] - kf.log['bg'][ii,:], alpha=0.6)
        axes3[ii].plot(sim.log['t'],  kf.log['sigma'][ii+3,:] * 180/np.pi, alpha=0.6, label='1-sigma', c='k')
        axes3[ii].plot(sim.log['t'], -kf.log['sigma'][ii+3,:] * 180/np.pi, alpha=0.6, c='k')
        axes3[ii].grid(True)

        axes4[ii].plot(sim.log['t'], sim.log['bg'][ii,:], alpha=0.6, label='sim')
        axes4[ii].plot(sim.log['t'], kf.log['bg'][ii,:], alpha=0.6, label='qekf')


    fig1.suptitle("Estimation error: attitude")
    axes1[0].legend()
    axes1[0].set_ylabel('ex [d]')
    axes1[1].set_ylabel('ey [d]')
    axes1[2].set_ylabel('ez [d]')
    axes1[2].set_xlabel("time [s]")

    fig2.suptitle("Angular velocity measurements")
    axes2[0].set_ylabel('wx [r/s]')
    axes2[1].set_ylabel('wy [r/s]')
    axes2[2].set_ylabel('wz [r/s]')
    axes2[2].set_xlabel("time [s]")

    fig3.suptitle("Estimation error: bias")
    axes3[0].set_ylabel('bgx [r/s])')
    axes3[1].set_ylabel('bgy [r/s])')
    axes3[2].set_ylabel('bgz [r/s])')
    axes3[2].set_xlabel("time [s]")

    fig4.suptitle("Simulated versus estimated bias")
    axes4[0].legend()
    axes4[0].set_ylabel('bgx [r/s])')
    axes4[1].set_ylabel('bgy [r/s])')
    axes4[2].set_ylabel('bgz [r/s])')
    axes4[2].set_xlabel("time [s]")
    
    plt.show()
    
    
