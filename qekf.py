# Need to import local pyquat, installed version not currently working
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyquat')))
import pyquat as pq
import pyquat.wahba as wahba
import pyquat.wahba.qmethod as qmethod

import numpy as np
import scipy.linalg as spl

class Qekf(object):
    tau     = 5400.0
    q_w_psd = 5.4154e-10

    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(6)
        self.q = pq.identity()
        self.P = np.zeros((6,6))
        for ii in range(0,3):
            self.P[ii,ii] = np.pi**2
        for ii in range(3,6):
            self.P[ii,ii] = 1e-4

        self.time = 0.0

        # As we propagate, log data
        self.log = { 't':     [0.0],
                     'wm':    [np.zeros(3)],
                     'qIB':   [pq.identity()],
                     'bg':    [np.zeros(3)],
                     'Pdiag': [np.zeros(6)] }

    @property
    def Pqq(self):
        return self.P[0:3,0:3]

    @property
    def Pqb(self):
        return self.P[0:3,3:]

    @property
    def Pbq(self):
        return self.P[3:,0:3]

    @property
    def Pbb(self):
        return self.P[3:,3:]

    @property
    def T(self):
        return self.q.to_matrix()

    def K(self, H, R):
        P11 = self.Pqq
        P21 = self.Pbq
        H1  = H[0:3,0:3]

        HT_HPH_plus_R = H1.T.dot( H1.dot(P11.dot(H1.T)) + R )
        Kq  = P11.dot(HT_HPH_plus_R)
        Kb  = P21.dot(HT_HPH_plus_R)

        return np.vstack((Kq, Kb))
        
        
    def update_attitude(self, *args):
        """Accept observations and reference vectors and use them to update
        the filter."""
        
        # Get prior information for qmethod
        Nqq_prior = spl.inv(self.Pqq)

        sigma_y = [1e-2, 1e-2]
        sigma_n = [1e-3, 1e-3]
        w = qmethod.compute_weights(sigma_y, sigma_n)

        y = np.vstack(args[:2]).T
        n = np.vstack(args[2:]).T
        q_post = qmethod.qmethod(y, n, w, q_prior = self.q, N_prior = Nqq_prior)

        T = self.T
        
        # Compute the covariance of the vectors which are orthogonal to
        # the observations and references.
        Rzz = qmethod.qekf_measurement_covariance(T, y, n, w, sigma_y, sigma_n)

        # Compute measurement model for attitude measurement
        Htheta = qmethod.qekf_measurement_model(T, y, n, w)
        
        Xi_prior = self.q.big_xi()

        # 4. Calculate attitude covariance posterior using Joseph form: eq 68
        Ktheta = spl.inv(-Nqq_prior*2 + Htheta)
        I_minus_KH = np.identity(3) - Ktheta.dot(Htheta)
        Pqq_post = I_minus_KH.dot(self.Pqq).dot(I_minus_KH.T) + Ktheta.dot(Rzz).dot(Ktheta.T)

        # 5. Update nonattitude states
        b_post = self.x[3:] + self.Pbq.dot(Nqq_prior).dot(Xi_prior.T.dot(q_post.to_vector().reshape((4)))) * 2

        # 6. Calculate total covariance update: eqs 70, 71
        Pbq_post = self.Pbq.dot(Nqq_prior).dot(Pqq_post)
        Pbb_post = self.Pbb + self.Pbq.dot( Nqq_prior.dot(Pqq_post).dot(Nqq_prior) - Nqq_prior ).dot(self.Pqb)
        Pqb_post = Pbq_post.T

        # Perform the update in preparation for the next 
        self.P = np.vstack(( np.hstack((Pqq_post, Pqb_post)),
                             np.hstack((Pbq_post, Pbb_post)) ))
        self.q = q_post
        self.x[0:3] = np.zeros(3)
        self.x[3:]  = b_post
        print("q = {}".format(self.q))
        print("Pqq = {}".format(self.P[0:3,0:3]))


    def dynamics_matrix(self, w):
        F = np.zeros((6,6))
        F[0:3,0:3] = -pq.skew(w)
        F[0:3,3:6] = -np.identity(3)
        F[3:6,3:6] = -np.identity(3)/self.tau
        return F

    def state_transition_matrix(self, w):
        return np.identity(6) + self.dynamics_matrix(w) * self.dt

    def process_noise(self):
        q_w = self.q_w_psd * self.dt
        return np.diag([0.0, 0.0, 0.0,
                        q_w, q_w, q_w])

    def propagate(self, w):
        """Advance the state forward in time using measurements from the
        gyroscope.

        """
        q_next = pq.propagate(self.q, w, self.dt)

        Phi = self.state_transition_matrix(w)
        P   = self.P
        Q   = self.process_noise()

        self.P = Phi.dot(P.dot(Phi.T)) + Q
        self.q = q_next
        # There are no dynamics on the state currently.

        self.time += self.dt
        
        self.log['t'].append(self.time)
        self.log['qIB'].append(q_next)
        self.log['wm'].append(w)
        self.log['bg'].append(self.x[3:6])
        self.log['Pdiag'].append(np.diag(self.P))

    def finish(self):
        """Close out logs"""
        self.log['t'] = np.hstack(self.log['t'])
        self.log['wm'] = np.vstack(self.log['wm']).T
        self.log['bg'] = np.vstack(self.log['bg']).T
        self.log['Pdiag'] = np.vstack(self.log['Pdiag']).T
        

if __name__ == '__main__':

    import pyquat.random as pqr
    import numpy.random as npr

    q_inrtl_to_body = pq.Quat(1.0, -2.0, 3.0, 4.0).normalized()
    print("q_ib = {}".format(q_inrtl_to_body))

    ref_misalign     = npr.randn(3) * 1e-6
    sun_obs_misalign = npr.randn(3) * 1e-5
    mag_obs_misalign = npr.randn(3) * 1e-5
    
    T_ref_err = np.identity(3) #- pq.skew(ref_misalign)
    T_sun_obs_err = np.identity(3) - pq.skew(sun_obs_misalign)
    T_mag_obs_err = np.identity(3) - pq.skew(mag_obs_misalign)
    
    mag_truth = np.array([0.0, 0.1, 1.0])
    mag_truth /= spl.norm(mag_truth)

    sun_truth = np.array([0.5, 0.5, 0.02])
    sun_truth /= spl.norm(sun_truth)

    Tib = q_inrtl_to_body.to_matrix()

    mag_ref = T_ref_err.dot(mag_truth)
    mag_ref /= spl.norm(mag_ref)
    sun_ref = T_ref_err.dot(sun_truth)
    sun_ref /= spl.norm(sun_ref)

    mag_obs  = T_mag_obs_err.dot(Tib.dot(mag_ref))
    mag_obs /= spl.norm(mag_obs)
    sun_obs  = T_sun_obs_err.dot(Tib.dot(sun_ref))
    sun_obs /= spl.norm(sun_obs)
    
    kf = Qekf(0.1)
    kf.update_attitude(sun_obs, mag_obs, sun_ref, mag_ref)
