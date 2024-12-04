from raft.raft_member import Member
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np

fname_design ="designs/OC4semi.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    # dict = design["members"][0]
    dict = design["platform"]["members"][1]
    # dict = design["platform"]["members"][2]
    mem_list = []

    mem_list.append(Member(dict, 1, heading=60))
    mem_list.append(Member(dict, 1, heading=180))
    mem_list.append(Member(dict, 1, heading=300))
    dict = design["platform"]["members"][0]
    mem_list.append(Member(dict, 1, heading=0))

    
for mem in mem_list:
    mem.setPosition()


from meshmagick.mesh import Mesh, Plane
from meshmagick.mesh_clipper import MeshClipper
from meshmagick.mmio import write_PNL

mesh = platformMesh(mem_list, sizeMax=1.0, constraint=0.25, recombined=True, clip=True)
mesh.show()

mesh1 = Mesh(vertices=mesh.getVertices(), faces=mesh.getFaces())
mesh1.heal_mesh()

print(mesh1.nb_vertices, mesh1.nb_faces)
# mesh1_sym = MeshClipper(mesh1, Plane((0,1,0))).clipped_mesh
# mesh1_sym.heal_mesh()
# mesh1_sym.show()
# print(mesh1_sym.nb_vertices, mesh1_sym.nb_faces)
# mesh1.show()

# tower_mesh = platformMesh([tower], sizeMax=1.0, constraint=0.25, recombined=True)
# tower_mesh.show()

write_PNL(f'models/HAMS/OC4_{mesh1.nb_faces}.dat', mesh1.vertices, mesh1.faces)