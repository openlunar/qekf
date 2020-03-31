import numpy as np

import qekf # import before pyquat
import pyquat as pq

from frames import planetodetic_to_pcpf
from frames import compute_T_pcpf_to_enu
from frames import compute_T_inrtl_to_pcpf

from qekf import Qekf

class SimFlatsat(object):
    def __init__(self,
                 lat = 37.7749 * np.pi/180.0,
                 lon = -122.41 * np.pi/180.0,
                 T_enu_to_body = np.identity(3)):

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

        self.log = {'t':   [self.t],
                    'qIB': [pq.from_matrix(self.T_inrtl_to_body)],
                    'rI': [self.r_inrtl],
                    'wBI': [np.zeros(3)]}

    @property
    def T_inrtl_to_body(self):
        return self.T_enu_to_body.dot(self.T_pcpf_to_enu.dot(self.T_inrtl_to_pcpf))

    def run(self, duration, dt = 0.1):
        start = self.t
        self.t += dt

        # Create qEKF
        kf = Qekf(dt)
        
        while self.t < start + duration:
            self.T_inrtl_to_pcpf = compute_T_inrtl_to_pcpf(self.t - start,
                                                           self.w_pcpf_wrt_inrtl[2])
            self.r_inrtl = self.T_inrtl_to_pcpf.T.dot(self.r_pcpf)
            self.T_pcpf_to_enu = compute_T_pcpf_to_enu(self.r_pcpf)

            w_body_wrt_inrtl_inrtl = self.w_pcpf_wrt_inrtl + self.w_body_wrt_pcpf

            T_inrtl_to_body        = self.T_inrtl_to_body
            w_body_wrt_inrtl_body  = T_inrtl_to_body.dot(w_body_wrt_inrtl_inrtl)
            # The above is the gyroscope measurement.

            kf.propagate(w_body_wrt_inrtl_body)

            self.log['t'].append(self.t)
            self.log['qIB'].append(pq.Quat.from_matrix(T_inrtl_to_body))
            #self.log['rI'].append(self.r_inrtl)
            self.log['wBI'].append(w_body_wrt_inrtl_body)
            

            self.t += dt
        
        
        # Finish up by stacking logs

        self.log['t'] = np.hstack(self.log['t'])
        #self.log['rI'] = np.vstack(self.log['rI']).T
        self.log['wBI'] = np.vstack(self.log['wBI']).T

        return kf

if __name__ == '__main__':

    sim = SimFlatsat()
    kf = sim.run(60.0)
    kf.finish()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True)
    
    for ii in range(0, 3):
        axes[ii].plot(sim.log['t'], sim.log['wBI'][ii,:], label='sim', alpha=0.6)
        axes[ii].plot(kf.log['t'],  kf.log['wm'][ii,:],  label='qekf', alpha=0.6)
        axes[ii].grid(True)

    axes[0].legend()
    axes[0].set_ylabel('wx [r/s]')
    axes[1].set_ylabel('wy [r/s]')
    axes[2].set_ylabel('wz [r/s]')
    axes[2].set_xlabel("time (s)")

    plt.show()
    
    
