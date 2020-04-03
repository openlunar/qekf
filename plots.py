import numpy as np

import matplotlib.pyplot as plt

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


def plot_attitude_errors(sim, kf):
    """Plot the delta between the simulated and estimated attitude (the
    attitude error) over time.
    
    Args:
        sim   simulation
        kf    qEKF

    Returns:
        A tuple consisting of the figure and the axes.
    """
    fig, axes = plt.subplots(3, 1, sharex = True)

    dq = dq_array(kf.log['qIB'], sim.log['qIB'])

    for ii in range(0, 3):
        axes[ii].plot(sim.log['t'], dq[ii,:] * 180/np.pi, label='attitude error', alpha=0.6)
        axes[ii].plot(sim.log['t'],  kf.log['sigma'][ii,:] * 180/np.pi, alpha=0.6, label='1-sigma', c='k')
        axes[ii].plot(sim.log['t'], -kf.log['sigma'][ii,:] * 180/np.pi, alpha=0.6, c='k')
        
        axes[ii].grid(True)
        

    fig.suptitle("Estimation error: attitude")
    axes[0].legend()
    axes[0].set_ylabel('ex [d]')
    axes[1].set_ylabel('ey [d]')
    axes[2].set_ylabel('ez [d]')
    axes[2].set_xlabel("time [s]")

    return fig, axes

def plot_angular_velocity_measurement_check(sim, kf):
    """Plot the delta between the measurements received by the qEKF and
    those sent by the simulation over time (should be 0).
    
    Args:
        sim   simulation
        kf    qEKF

    Returns:
        A tuple consisting of the figure and the axes.

    """
    
    fig, axes = plt.subplots(3, 1, sharex = True)

    for ii in range(0, 3):
        axes[ii].plot(sim.log['t'], sim.log['wBI'][ii,:] - kf.log['wm'][ii,:], label='delta', alpha=0.6)
        #axes[ii].plot(kf.log['t'],  kf.log['wm'][ii,:],  label='qekf', alpha=0.6)
        axes[ii].grid(True)

    fig.suptitle("Angular velocity measurement difference")
    axes[0].set_ylabel('wx [r/s]')
    axes[1].set_ylabel('wy [r/s]')
    axes[2].set_ylabel('wz [r/s]')
    axes[2].set_xlabel("time [s]")

    return fig, axes

def add_bias_labels(fig, axes):
    axes[0].legend()
    axes[0].set_ylabel('bgx [r/s])')
    axes[1].set_ylabel('bgy [r/s])')
    axes[2].set_ylabel('bgz [r/s])')
    axes[2].set_xlabel("time [s]")

def plot_gyroscope_bias_errors(sim, kf):
    """Plot the delta between the simulated and estimated gyroscope bias
    over time.
    
    Args:
        sim   simulation
        kf    qEKF

    Returns:
        A tuple consisting of the figure and the axes.

    """
    fig, axes = plt.subplots(3, 1, sharex = True)

    for ii in range(0, 3):
        axes[ii].plot(sim.log['t'], sim.log['bg'][ii,:] - kf.log['bg'][ii,:], alpha=0.6)
        axes[ii].plot(sim.log['t'],  kf.log['sigma'][ii+3,:] * 180/np.pi, alpha=0.6, label='1-sigma', c='k')
        axes[ii].plot(sim.log['t'], -kf.log['sigma'][ii+3,:] * 180/np.pi, alpha=0.6, c='k')
        axes[ii].grid(True)

    fig.suptitle("Estimation error: gyroscope bias")
    add_bias_labels(fig, axes)

    return fig, axes

def plot_gyroscope_bias_check(sim, kf):
    """Plot the simulated and estimated gyroscope biases over time against
    one another.
    
    Args:
        sim   simulation
        kf    qEKF

    Returns:
        A tuple consisting of the figure and the axes.

    """
    
    fig, axes = plt.subplots(3, 1, sharex = True)

    for ii in range(0, 3):
        axes[ii].plot(sim.log['t'], sim.log['bg'][ii,:], alpha=0.6, label='sim')
        axes[ii].plot(sim.log['t'], kf.log['bg'][ii,:], alpha=0.6, label='qekf')

    fig.suptitle("Simulated versus estimated gyroscope bias")
    add_bias_labels(fig, axes)

    return fig, axes
