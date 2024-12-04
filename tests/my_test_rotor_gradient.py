import pytest
import numpy as np
# from numpy.testing import assert_allclose
import yaml
import pickle
import raft
from raft.raft_rotor import Rotor
from raft.helpers import getFromDict
import os

raft_dir = ''
fname_design = os.path.join(raft_dir,'designs/OC4semi.yaml')

# open the design YAML file and parse it into a dictionary for passing to raft
with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)
    # transfer some dictionary contents that would normally be done higher up in RAFT
    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']

    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0

UU = np.arange(6,17,2)           # wind speeds
ws = np.arange(0.01,6.0,0.01) # frequencies (rad/s)
# make Rotor object
rotor = Rotor(design['turbine'], ws, 0)

#
ccblade = rotor.ccblade
ccblade.derivatives = True

Uinf = 4
Omega = 10
pitch = 0
azimuth = np.rad2deg(0)

loads, derives = ccblade.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
Np, Tp = (loads["Np"], loads["Tp"])

loads, derives = ccblade.evaluate(Uinf, Omega, pitch)

a1 = 0


