from raft.raft_member_yang import Member
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np

fname_design ="examples/OC4semi-RAFT_QTF.yaml"
# fname_design ="../designs/OC4semi.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    # dict = design["members"][0]
    dict = design["platform"]["members"][1]
    # dict = design["platform"]["members"][2]
    mem_list = []
    attach_list = []
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    dict = design["platform"]["members"][0]
    mem_list.append(Member(dict, 1, heading=0))

Cmat = np.zeros((6,6))
Cmat_orig = np.zeros((6,6))

for mem in mem_list:

    mem.setPosition()

    Cmat += mem.getHydrostaticsFromMesh()[1]
    Cmat_orig += mem.getHydrostatics()[1]

from tabulate import tabulate

print(tabulate(Cmat, headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))
print(tabulate(Cmat_orig, headers=["x", "y", "z", "xx", "yy", "zz"], floatfmt=".3f"))