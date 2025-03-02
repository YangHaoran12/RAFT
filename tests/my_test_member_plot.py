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

mem = Member(dict, 1)

Surge = 0   #[m]
Sway  = 0   #[m]
Heave = 0  #[m]
Roll  = 0   #[deg]
Pitch = 0   #[deg]
Yaw   = 0   #[deg]

r6=[Surge, Sway, Heave, np.deg2rad(Roll), np.deg2rad(Pitch), np.deg2rad(Yaw)]

mem.setPosition()
r1 = mem.getHydrostaticsOld()
r2 = mem.getHydrostatics()
r3 = mem.getHydrostaticsFromMesh()

ax = plt.figure().add_subplot(projection='3d')

from tabulate import tabulate

print(tabulate(r1[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r2[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r3[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print('-------Fvec---------')
print(tabulate(np.array([r1[0],r2[0],r3[0]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))
print('--------r_center----------')
print(tabulate(np.array([r1[3],r2[3],r3[3]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))
mem.plot(ax)
print('-----------------------------------')

mem.setPosition(r6)
r1 = mem.getHydrostaticsOld(r6[:3])
r2 = mem.getHydrostatics(r6[:3])
r3 = mem.getHydrostaticsFromMesh(r6[:3])


from tabulate import tabulate

print(tabulate(r1[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r2[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r3[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))

print('-------Fvec---------')
print(tabulate(np.array([r1[0],r2[0],r3[0]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))
print('--------r_center----------')
print(tabulate(np.array([r1[3],r2[3],r3[3]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))
mem.plot(ax)
plt.axis('equal') 
plt.show()



