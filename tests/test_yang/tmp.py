import yaml
from raft.raft_member import Member

import numpy as np

from raft.member2pnl import meshMember
from meshmagick.mesh import Mesh, Plane

import raft.member2Gmsh as m2g
# fname_design = "tests/test_data/mem_srf_vert_circ_cyl.yaml"
# fname_design ="tests/test_data/mem_srf_vert_rect_cyl.yaml"
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
fname_design ="examples/OC4semi-RAFT_QTF.yaml"
# fname_design ="tests/test_data/mem_subm_horz_rect_cyl.yaml"

fname_design = f"tests/test_data/{list_files[6]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["members"][0]
# dict = design["platform"]["members"][1]
mem = Member(dict, 1)

mem.setPosition()

import copy

tmp = copy.copy(mem)

a = 1

# vertices, faces1 = meshMember(mem.stations, mem.d, mem.rA, mem.rB)

# faces = []

# for i in faces1:
#     if i[1] == 3:
#         faces.append([i[2], i[3], i[4], i[2]])
#     elif i[1] == 4:
#         faces.append([i[2], i[3], i[4], i[5]])

# faces = np.asarray(faces, dtype=int) - 1

# mesh = Mesh(vertices, faces, "ver_cyl")
# mesh.quick_save()

# a = 1

# if mem.shape == "rectangular":
#     m2g.memberMeshRect(mem.stations, mem.sl, mem.R, index=0)
# elif mem.shape == "circular":
#     m2g.memberMeshCirc(mem.stations, mem.d, mem.rA, mem.rB, index=0)



# for i in faces1:
# if i[1] == 3:
#     faces.append([i[2], i[3], i[4], i[2]])
# elif i[1] == 4:
#     faces.append([i[2], i[3], i[4], i[5]])
# faces = np.asarray(faces, dtype=int) - 1

# mesh = Mesh(vertices, faces, "ver_cyl")
# mesh.quick_save()


# m2g.writeMesh(visual=True, meshFile="mem_subm_horz_rect_cyl.msh", hydroFlag=False)

import meshmagick.mmio as mmio

vertices, faces = mmio.load_MSH22("LHEEA/mem_subm_horz_rect_cyl.msh")

mesh = Mesh(vertices, faces)

plane = Plane(scalar=10)

from meshmagick.mesh_clipper import MeshClipper 

mc = MeshClipper(mesh, plane, verbose=True)
aa = mc.lower_mesh

aa.quick_save()


a = 1