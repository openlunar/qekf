import numpy as np
import scipy.linalg as spl

def planetodetic_to_pcpf(lon, lat,
                         h = 0.0,
                         a = 6378137.0,
                         b = 6356752.314245):
    """Convert planetodetic coordinates (longitude, latitude, and height
    above the ellipsoid) into planet-centered, planet-fixed
    coordinates.

    By default, this uses the WGS84 ellipsoid and a height of 0.

    Reference:

        [0] https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

    Args:
        lon  longitude (radians)
        lat  latitude (radians)
        h    altitude (m)
        a    ellipsoid semimajor axis (m)
        b    ellipsoid semiminor axis (m)

    Returns:
        A numpy array consisting of x, y, and z coordinates in a
    planet-centered, planet-fixed frame.

    """
    b2_over_a2 = b**2 / a**2

    # Compute prime vertical radius of curvature
    e2 = 1.0 - b2_over_a2
    N = a / np.sqrt(1.0 - e2 * np.sin(lat))
    
    return np.array([ (N + h) * np.cos(lat) * np.cos(lon),
                      (N + h) * np.cos(lat) * np.sin(lon),
                      (b2_over_a2 * N + h) * np.sin(lat) ])


def compute_T_pcpf_to_enu(r_pcpf):
    """Find the attitude of an East--North--Up frame located at a given
    location in a planet-centered, planet-fixed frame relative to that
    frame.

    Args:
        r_pcpf   planet-centered, planet-fixed position (m)

    Returns:
        A 3x3 transformation matrix.

    """
    up     = r_pcpf / spl.norm(r_pcpf)
    z      = np.array([0.0, 0.0, 1.0])
    east   = np.cross(up, z)
    east  /= spl.norm(east)
    north  = np.cross(up, east)
    north /= spl.norm(north)

    return np.vstack((east, north, up)).T


def rotate_z(t):
    """Rotation about the z axis by an angle t.

    Args:
        t  angle (radians)

    Returns:
        3x3 orthonormal rotation matrix.
    """
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t),  np.cos(t), 0.0],
                     [0, 0, 1]])

def compute_T_inrtl_to_pcpf(dt,
                            w_pcpf           = 2.0 * np.pi / (23 * 3600.0 + 56 * 60.0 + 4.091),
                            T_inrtl_to_pcpf0 = None):
    """Compute the approximate attitude of the planet-centered,
    planet-fixed frame relative to a planet-centered inertial frame.

    Args:
        dt                 time since epoch (s)
        w_pcpf             angular rate of the planet about its axis
                           (r/s; defaults to the inverse of earth's 
                           sidereal day length times 2*pi)
        T_inrtl_to_pcpf0   attitude matrix at epoch (defaults to
                           identity matrix)

    Returns:
        A 3x3 transformation matrix.
    """
    if T_inrtl_to_pcpf0 is None:
        if dt == 0:
            return np.identity(3)
        return rotate_z(w_pcpf * dt)
    else:
        if dt == 0:
            return T_inrtl_to_pcpf0
        return rotate_z(w_pcpf * dt).dot(T_inrtl_to_pcpf0)
    
    
