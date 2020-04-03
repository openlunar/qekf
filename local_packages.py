# Need to import local pyquat, installed version not currently working
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyquat')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../WMM2015')))
