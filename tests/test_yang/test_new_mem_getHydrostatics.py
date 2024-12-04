from raft.raft_member_yang import Member
import yaml
import matplotlib.pyplot as plt
import numpy as np

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
    ]

fname_design = f"../tests/test_data/{list_files[5]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["members"][0]
# dict = design["platform"]["members"][1]

mem = Member(dict, 1)

# mem.setPosition(r6=[1,1,0,0.01*np.pi,0.01*np.pi,0])
mem.setPosition()
r1 = mem.getHydrostatics()
r2 = mem.getHydrostaticsYang()
r3 = mem.getHydrostaticsFromMesh()

from tabulate import tabulate

print(tabulate(r1[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r2[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(r3[1], headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))

print(tabulate(np.array([r1[0],r2[0],r3[0]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))

# print(tabulate(np.array([r1[0],r2[0],r3[0]]).T, headers=["Orig", "Modify", "Gmsh"], floatfmt=".3f"))

# mem_list = []

# mem_list.append(Member(dict,1, heading=60))
# mem_list.append(Member(dict,1, heading=180))
# mem_list.append(Member(dict,1, heading=300))

# Fvec = np.zeros(6)
# for mem in mem_list:
#     mem.setPosition()
#     Fvec += mem.getHydrostaticsFromMesh()[0]


# print(tabulate(np.array([r1[0],r2[0],r3[0], Fvec]).T, headers=["Orig", "Modify", "Gmsh", "All"], floatfmt=".3f"))

# print(f"original method: \n Awp={r1[4]}")
# print(f"modified method: \n Awp={r2[4]}")
# print(f"modified method: \n Awp={r3[4]}")

# print(f"original method: \n Iwp={r1[5]}")
# print(f"modified method: \n Iwp={r2[5]}")
# print(f"modified method: \n Awp={r3[5]}")

# print(f"original method: \n Fvec={r1[0]}")
# print(f"modified method: \n Fvec={r2[0]}")
# print(f"modified method: \n Fvec={r3[0]}")


# mem.getInertia()

# M_struc, mass, center = mem.getMemberInertia()

# print(f"Mass matrix: \n {M_struc}")
# print(f"Mass       : \n {mass}")
# print(f"Center     : \n {center}")


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# mem.plotSurface(ax, nodes=2)

# plt.show()