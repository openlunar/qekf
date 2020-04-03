import local_packages

import pyquat as pq
import pyquat.wahba as wahba
import pyquat.wahba.qmethod as qmethod

import numpy as np
import scipy.linalg as spl

class Qekf(object):
    tau     = 5400.0     # time constant on the attitude bias, which
                         # is an exponentially-correlated random
                         # variable (ECRV)
    q_w_psd = 5.4154e-10 # power spectral density on the noise coming
                         # out of the gyroscope
    q_w_psd_tuning_factor = 1.0

    def __init__(self, dt,
                 q_inrtl_to_body_init = pq.identity()):
        """Constructor for the q-method extended Kalman filter.

        Args:
            dt                    propagation time step (s)
            q_inrtl_to_body_init  initial attitude guess (of type
                                  pyquat.Quat; defaults to pq.identity())

        """
        self.dt = dt
        self.time = 0.0

        # Setup state estimate
        self.x = np.zeros(6)
        self.q = q_inrtl_to_body_init # attitude estimate

        # Setup and initialize covariance
        self.P = np.zeros((6,6))
        for ii in range(0,3):
            self.P[ii,ii] = (0.5 * np.pi)**2
        for ii in range(3,6):
            self.P[ii,ii] = 1e-8

        # As we propagate, log data
        self.log = { 't':     [0.0],
                     'wm':    [np.zeros(3)],
                     'qIB':   [self.q],
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
        
        
    def update_attitude(self, *args,
                        sigma_y = [1e-3,  1e-3],
                        sigma_n = [1e-11, 1e-3],
                        w       = None):
        """Accept observations and reference vectors and use them to update
        the filter.

        This method currently accepts four unnamed arguments, all of
        which are length-3 vectors. The first two are the observed
        vectors (e.g. to the sun and in the direction of the magnetic
        field). The second two are the reference vectors (where the
        model suggests these should be pointed).

        The method, when complete, has updated the filter's estimate
        of the attitude, the state estimate, and the covariance.

        Args:
            obs1   unit vector observation 1
            obs2   unit vector observation 2
            ref1   unit reference vector corresponding to obs1
            ref2   unit reference vector corresponding to obs2
        
        Kwargs:
            sigma_y  standard deviations for obs1 and obs2 in a list,
                     tuple, or numpy array
            sigma_n  standard deviations of ref1 and ref2 in a list,
                     tuple, or numpy array
            w        weights list, tuple, or numpy array for each
                     vector, which may be provided in lieu of sigma_y
                     and sigma_n (default: None)

        Returns:
            A tuple of the state update and the delta quaternion
        reflecting the change in attitude.

        """
        
        # Get prior information for qmethod
        Nqq_prior = spl.inv(self.Pqq)

        if w is None:
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
        db = self.Pbq.dot(Nqq_prior).dot(Xi_prior.T.dot(q_post.to_vector().reshape((4)))) * 2

        # 6. Calculate total covariance update: eqs 70, 71
        Pbq_post = self.Pbq.dot(Nqq_prior).dot(Pqq_post)
        Pbb_post = self.Pbb + self.Pbq.dot( Nqq_prior.dot(Pqq_post).dot(Nqq_prior) - Nqq_prior ).dot(self.Pqb)
        Pqb_post = Pbq_post.T

        # Perform the update in preparation for the next 
        self.P = np.vstack(( np.hstack((Pqq_post, Pqb_post)),
                             np.hstack((Pbq_post, Pbb_post)) ))

        dq_body = q_post * self.q.conjugated()
        self.q = q_post

        dx = np.hstack((np.zeros(3), db))
        self.x += dx
        print("q = {}".format(self.q))
        print("Pqq = {}".format(self.P[0:3,0:3]))

        return dx, dq_body


    def dynamics_matrix(self, omega):
        """Linearize the dynamics at the current time point, using the current
        IMU measurements and the state.

        Args:
            omega   angular velocity measurement (see propagate())

        Returns:
            A square matrix of the same size as the state vector.

        """
        F = np.zeros((6,6))
        F[0:3,0:3] = -pq.skew(omega)
        F[0:3,3:6] = -np.identity(3)
        F[3:6,3:6] = -np.identity(3)/self.tau
        return F

    def state_transition_matrix(self, omega):
        """Compute \Phi(t_{k}, t_{k-1}) using IMU measurements and aspects of
        the filter state.

        This uses the first-order approximation

            \Phi \approx I + F|_{t_k} \Delta t

        where t_k is the current time, and F is the Jacobian of the
        state dynamics (from dynamics_matrix()).

        Args:
            omega  angular velocity measurement (as passed to propagate())

        Returns:
            An NxN matrix, where N is the size of the state vector.

        """
        Phi = np.identity(6) + self.dynamics_matrix(omega) * self.dt
        Phi[3:6,3:6] = np.identity(3) * np.exp(-self.dt / self.tau)
        return Phi
        
    def process_noise(self):
        """Compute the Q matrix.

        Returns:
            A diagonal matrix the same size as the covariance P.

        """
        q_w = self.q_w_psd * self.q_w_psd_tuning_factor * self.dt
        return np.diag([0.0, 0.0, 0.0, q_w, q_w, q_w])

    def propagate(self, omega):
        """Advance the state forward in time using measurements from the
        gyroscope.

        Args:
            omega   measured angular velocity (body with respect to the
                    inertial frame, expressed in the body frame)

        """
        q_next = pq.propagate(self.q, omega, self.dt)

        Phi = self.state_transition_matrix(omega)
        P   = self.P
        Q   = self.process_noise()

        self.P = Phi.dot(P.dot(Phi.T)) + Q
        self.q = q_next
        # There are no dynamics on the state currently.

        self.time += self.dt
        
        self.log['t'].append(self.time)
        self.log['qIB'].append(q_next)
        self.log['wm'].append(omega)
        self.log['bg'].append(np.array(self.x[3:6])) # copy! don't ref
        self.log['Pdiag'].append(np.diag(self.P))

    def finish(self):
        """Close out logs.

        """
        self.log['t'] = np.hstack(self.log['t'])
        self.log['wm'] = np.vstack(self.log['wm']).T
        self.log['bg'] = np.vstack(self.log['bg']).T
        self.log['sigma'] = np.sqrt(np.vstack(self.log['Pdiag']).T)
        del self.log['Pdiag']
        

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
