import local_packages

import numpy as np
import numpy.random as npr
import scipy.linalg as spl

import pyquat as pq

from spice_loader import SpiceLoader
from spiceypy import spiceypy as spice

import wmm2015 # see local_packages

def lookup_u_sun_inrtl(t):
    """Return the unit vector to the sun in an earth-centered inertial
    frame.

    Uses SPICE and treats t as an ephemeris time.

    Currently, this provides an approximation, looking up the sun from
    the center of the earth rather than the sensor location.

    Args:
        t   ephemeris time (s)
    
    Returns:
        A unit vector pointing at the sun.
    """
    r = spice.spkezp(10, t, 'J2000', 'LT', 399)[0]
    return r / spl.norm(r)


def lookup_u_mag_enu(lon, lat,
                     alt  = 0.0,
                     year = 2020):
    """Return the unit vector of the magnetic field in a local ENU
    frame.

    Args:
        lon   longitude of measurement (rad)
        lat   planetodetic/geodetic latitude of measurement (rad)
        alt   altitude above geoid (m; defaults to 0)
        year  decimal year of measurement (default is 2020)

    Returns:
        A unit vector in an ENU frame.
    """

    # WMM takes degrees and km
    mag = wmm2015.wmm(lat * 180/np.pi, lon * 180/np.pi, alt / 1000.0, year)

    # Get vector in north/east/down coordinates.
    enu = np.array([ mag.east.item(),
                     mag.north.item(),
                    -mag.down.item()  ])
    return enu / spl.norm(enu)


def noisify_line_of_sight_vector(u, sigma):
    """Add misalignment to a line-of-sight unit vector measurement.

    Args:
        u      unit vector measurement (length 3)
        sigma  standard deviation, either scalar or vector (length 3)

    Returns:
        A noisy unit vector (length 3).
    """
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

def gravity(t, r, mu = 3.986004418e14):
    """Compute acceleration due to gravity given a point mass."""
    r2 = r.T.dot(r)
    r1 = np.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    if r2 < SMALL:
        raise ValueError("position for gravity calculation too close to origin")

    return r * -mu / r3

