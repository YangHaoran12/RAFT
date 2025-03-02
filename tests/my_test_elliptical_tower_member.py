from raft.raft_member import Member, TowerMember
from raft.raft_rotor import raft_dir
import yaml
import matplotlib.pyplot as plt
import numpy as np

import os

list_files = [
    'mem_srf_vert_circ_cyl.yaml',    
    'mem_srf_vert_rect_cyl.yaml',
    'mem_srf_pitch_circ_cyl.yaml',
    'mem_srf_pitch_rect_cyl.yaml',
    'mem_srf_inc_circ_cyl.yaml',
    'mem_srf_inc_rect_cyl.yaml',
    'mem_subm_horz_circ_cyl.yaml',
    'mem_subm_horz_rect_cyl.yaml',
    'mem_srf_vert_tap_circ_cyl.yaml',
    'mem_srf_vert_tap_rect_cyl.yaml',
    'mem_srf_vert_ellipse_cyl.yaml'
    ]

fname_design = os.path.join(raft_dir, f"tests/test_data/{list_files[-1]}")

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["members"][0]
# dict = design["platform"]["members"][1]

Surge = 10   #[m]
Sway  = 0    #[m]
Heave = 0    #[m]
Roll  = 0    #[deg]
Pitch = 0    #[deg]
Yaw   = 0    #[deg]

r6=[Surge, Sway, Heave, np.deg2rad(Roll), np.deg2rad(Pitch), np.deg2rad(Yaw)]

mem_ellipse = TowerMember(dict, 1)
mem_ellipse.setPosition(r6)
mem_ellipse.getStripStructForce(g=9.81)
result_ellipse = mem_ellipse.getInertia(r6[:3])

dict['shape'] = 'circular'
dict['d'] = [9.85, 9.85]
mem = Member(dict, 1)
mem.setPosition(r6)
result1 = mem.getInertia(r6[:3])

from tabulate import tabulate

M_struc_ellipse = mem_ellipse.M_struc
M_struc = mem.M_struc

M_show = np.asarray([[M_struc_ellipse[i,i],M_struc[i,i]] for i in range(6)])

print(tabulate(M_show.T, headers=["m", "m", "m", "Ixx", "Iyy", "Izz"], floatfmt=".3f"))

F_struc = mem_ellipse.F_struc
print(tabulate(F_struc, headers=["Fx", "Fy", "Fz", "Mx", "My", "Mz"], floatfmt=".3f"))




