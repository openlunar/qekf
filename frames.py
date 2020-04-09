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
        a    ellipsoid semimajor axis / equatorial radius (m)
        b    ellipsoid semiminor axis / polar radius (m)

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

def compute_reduced_latitude(mu, f):
    """Compute the reduced latitude from the best available guess about
    the planetodetic latitude.

    This is a helper for pcpf_to_planetodetic().

    Args:
        mu   planetodetic latitude estimate (rad)
        f    first flattening

    Returns:
        The new estimate of the reduced latitude (rad).
    """
    return np.arctan((1.0 - f) * np.sin(mu) / np.cos(mu))

def compute_planetodetic_latitude(s, rz, a, f, e2, beta):
    """Compute the planetodetic latitude from the reduced latitude.

    This is a helper for pcpf_to_planetodetic().

    Args:
        s     distance from the polar axis in the x/y plane (m)
        rz    distance along polar axis (m)
        a     equatorial radius (m)
        f     first flattening
        e2    square of the first eccentricity
        beta  reduced latitude (rad)

    Returns:
        The planetodetic latitude (rad).
    """
    return np.arctan((rz + (e2 * (1.0 - f) * a * np.sin(beta)**3 / (1.0 - e2)) / (1.0 - e2)) / (s - e2 * a * np.cos(beta)**3))

def pcpf_to_planetodetic(r_pcpf,
                         a     = 6378137.0,
                         f     = 0.003352810664775694,
                         tol   = 1e-4,
                         small = 1e-12):
    """Given PCPF position information, compute the planetodetic/geodetic
    longitude, latitude, and altitude above the geoid.

    Args:
        r_pcpf  position in PCPF coordinates (m)
        a       semimajor axis / equatorial radius (m)
        f       first flattening
        tol     tolerance for latitude iteration and convergence (rad)

    Returns:
        A tuple of longitude, planetodetic latitude, and altitude. The
    first two are in radians and the third is in meters.
    """
    # Compute square of first eccentricity
    e2 = 1.0 - (1.0 - f)**2
    
    # Compute longitude and radius from polar axis
    if r_pcpf[0] < small:
        lon = 0.0
    else:
        lon = np.arctan(r_pcpf[1] / r_pcpf[0])
        
    s   = np.sqrt(r_pcpf[0]**2 + r_pcpf[1]**2)
    if s < small:
        if r_pcpf[2] > 0:
            beta = np.pi/2.0
        else:
            beta = -np.pi/2.0
    else:
        # Start by calculating an initial guess for the planetodetic latitude.
        beta     = np.arctan( r_pcpf[2] / ((1.0 - f) * s) )
    last_lat = compute_planetodetic_latitude(s, r_pcpf[2], a, f, e2, beta)

    # Re-calculate it using our value for mu
    beta     = compute_reduced_latitude(last_lat, f)
    lat      = compute_planetodetic_latitude(s, r_pcpf[2], a, f, e2, beta)
    
    while np.abs(lat - last_lat) > tol:
        # Re-calculate reduced latitude
        beta     = compute_reduced_latitude(lat, f)
        last_lat = lat
        lat      = compute_planetodetic_latitude(s, r_pcpf[2], a, f, e2, beta)

    # Having converged, we now compute the altitude above the geoid
    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2) # radius of curvature
                                               # in the vertical prime
    h = s * np.cos(lat) + (r_pcpf[2] + e2 * N * np.sin(lat)) * np.sin(lat) - N

    return lon, lat, h


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

    # The negative sign inside rotate_z controls the direction of
    # rotation.
    if T_inrtl_to_pcpf0 is None:
        if dt == 0:
            return np.identity(3)
        return rotate_z(-w_pcpf * dt)
    else:
        if dt == 0:
            return T_inrtl_to_pcpf0
        return rotate_z(-w_pcpf * dt).dot(T_inrtl_to_pcpf0)
    
    
