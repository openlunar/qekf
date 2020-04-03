import numpy as np

from spiceypy import spiceypy as spice
import spiceypy.utils.support_types as stypes

import os

class SpiceLoader(object):

    def __init__(self, mission = None, dir = 'kernels/', id = -5440):
        self.dir        = dir
        self.object_id  = id

        names = ['de432s.bsp', 'pck00010.tpc', 'naif0012.tls',
                 'moon_pa_de421_1900-2050.bpc', 'moon_080317.tf',
                 'moon_fixed_me.tf', 'gm_de431.tpc', 'earth_070425_370426_predict.bpc',
                 'earthstns_itrf93_050714.bsp', 'earth_topo_050714.tf']
        if mission is not None:
            names.append(mission + '.bsp')
            self.mission = mission
        else:
            self.mission = None

        self.load(names)

    def load(self, names):
        kernels_dir = os.path.join(os.path.dirname(__file__), self.dir)
        
        paths = []
        for name in names:
            paths.append(kernels_dir + name)
            
        spice.furnsh(paths)

        self.loaded = paths

        # Pre-cache some things that we'll almost certainly need access to.
        self.load_constants()
        
        return self.loaded

    def load_constants(self):
        self.r_earth         = spice.bodvcd(399, 'RADII', 3)[1]
        self.r_moon          = spice.bodvcd(301, 'RADII', 3)[1]
        self.mu_earth        = spice.bodvcd(399, 'GM', 1)[1]
        self.mu_moon         = spice.bodvcd(301, 'GM', 1)[1]

        # Constants that should ultimately probably go in SPICE kernels.
        # For now let's just pretend they're in SPICE kernels.
        self.T_body_to_att   = np.identity(3)
        self.T_body_to_cam   = np.identity(3)

        if self.mission is not None:
            self.start, self.end = self.coverage()

    def radii(self, body):
        if body in (399, 'earth'):
            return self.r_earth
        elif body in (301, 'moon'):
            return self.r_moon
        else:
            raise NotImplemented("body {} not pre-loaded".format(body))

    @classmethod
    def spk_coverage(self, path, id = -5440):
        """Get start and end ephemeris times for a SPK file."""
        coverage = stypes.SPICEDOUBLE_CELL(2)
        spice.spkcov(path, id, coverage)
        return spice.wnfetd(coverage, 0)

    def coverage(self, id = None):
        """Get mission coverage for a specific NAIF ID"""
        if id is None:
            id = self.object_id
        return SpiceLoader.spk_coverage(self.dir + self.mission + '.bsp', id)
