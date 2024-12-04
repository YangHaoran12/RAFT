from raft.raft_member_yang import TowerMember, Member
import yaml
import matplotlib.pyplot as plt
import numpy as np

# fname_design = "tests/test_data/mem_srf_vert_circ_cyl.yaml"
# fname_design ="tests/test_data/mem_subm_horz_rect_cyl.yaml"
# fname_design ="tests/test_data/mem_srf_inc_circ_cyl.yaml"
# fname_design ="examples/OC4semi-RAFT_QTF.yaml"
fname_design ="tests/test_data/mem_srf_vert_ellipse_cyl.yaml"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["members"][0]
# dict = design["platform"]["members"][1]

mem = TowerMember(dict, 1)

# mem.setPosition(r6=[1,1,0,0.01*np.pi,0.01*np.pi,0])
mem.setPosition()
mass, cog, mshell, _, _ = mem.getInertia()

import pprint
print(f"mass = {mass}")
print(f"cog = {cog}")

pprint.pprint(mem.M_struc)
A0 = mem.M_struc
# r1 = mem.getHydrostatics()
# r2 = mem.getHydrostaticsYang()

# print(f"original method: \n Cmat={r1[1]}")
# print(f"modified method: \n Cmat={r2[1]}")

# print(f"original method: \n Awp={r1[4]}")
# print(f"modified method: \n Awp={r2[4]}")

# print(f"original method: \n Iwp={r1[5]}")
# print(f"modified method: \n Iwp={r2[5]}")
dict['d'] = 9.85
dict['shape'] = 'circular'
mem = Member(dict, 1)
mem.setPosition()
mass, cog, mshell, _, _ = mem.getInertia()
print(f"mass = {mass}")
print(f"cog = {cog}")

pprint.pprint(mem.M_struc)
A1 = mem.M_struc
import numpy as np
print(f"Analytical surface area = {np.pi*dict['d']*(dict['stations'][1]-dict['stations'][0])}")
print(f"Analytical mass = {np.pi*dict['d']*dict['t']*dict['rho_shell']*(dict['stations'][1]-dict['stations'][0])}")

# pprint.pprint(f"diff = {np.divide(abs((A0-A1)), A0)}")
