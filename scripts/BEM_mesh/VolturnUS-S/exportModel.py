from raft.raft_member_yang import Member, TowerMember
from raft.raft_mesh import platformMesh, sPlatformMesh
from raft.raft_rotor_yang import raft_dir
import yaml
import numpy as np
import os

fname_design ="designs/VolturnUS-S.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)
    
    mem_list = []
    
    dict = design["platform"]["members"][1]
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    
    dict = design["platform"]["members"][0]
    mem_list.append(Member(dict, 1, heading=0))

    dict = design["platform"]["members"][2]
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))

    dict = design["platform"]["members"][3]
    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))

    dict = design['turbine']['tower']
    mem_list.append(TowerMember(dict, 1, heading=0))

    
for mem in mem_list:
    mem.setPosition()


from meshmagick.mesh import Mesh, Plane
from meshmagick.mmio import write_NEM, write_MAR, write_PNL

mesh = platformMesh(mem_list, sizeMax=1.6, constraint=0.25, recombined=True, clip=False)
mesh.show()
print(mesh.nb_vertices, mesh.nb_faces)

from meshmagick.mesh import Mesh

mesh1 = Mesh(vertices=mesh.getVertices(), faces=mesh.getFaces())
mesh1.heal_mesh()
# mesh1.show()

# tower_mesh = platformMesh([tower], sizeMax=1.0, constraint=0.25, recombined=True)
# tower_mesh.show()

meshFile = os.path.join(raft_dir, f'models/nemoh/VolturnUS-S_full.dat')

write_MAR(meshFile, mesh1.vertices, mesh1.faces)

