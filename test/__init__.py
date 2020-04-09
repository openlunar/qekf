import unittest
from test import test_qekf, test_sim_flatsat, test_frames

def qekf_test_suite():
    """
    Load unit tests from each file for automatic running using 
    `python setup.py test`.
    """
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_qekf))
    suite.addTests(loader.loadTestsFromModule(test_sim_flatsat))
    suite.addTests(loader.loadTestsFromModule(test_frames))

    return suite


