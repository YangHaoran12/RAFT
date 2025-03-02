from raft.raft_member import Member
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np

# fname_design ="examples/OC4semi-RAFT_QTF.yaml"
fname_design ="designs/OC4semi.yaml"

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

    # dict = design["turbine"]["tower"]
    # mem_list.append(Member(dict, 1))

    dict = design["platform"]["members"][2]
    attach_list.append(Member(dict, 1, heading=60))
    attach_list.append(Member(dict, 1, heading=180))
    attach_list.append(Member(dict, 1, heading=300))

    dict = design["platform"]["members"][3]
    attach_list.append(Member(dict, 1, heading=60))
    attach_list.append(Member(dict, 1, heading=180))
    attach_list.append(Member(dict, 1, heading=300))

    dict = design["platform"]["members"][4]
    attach_list.append(Member(dict, 1, heading=60))
    attach_list.append(Member(dict, 1, heading=180))
    attach_list.append(Member(dict, 1, heading=300))

    dict = design["platform"]["members"][5]
    attach_list.append(Member(dict, 1, heading=60))
    attach_list.append(Member(dict, 1, heading=180))
    attach_list.append(Member(dict, 1, heading=300))

    dict = design["platform"]["members"][6]
    attach_list.append(Member(dict, 1, heading=60))
    attach_list.append(Member(dict, 1, heading=180))
    attach_list.append(Member(dict, 1, heading=300))

    # dict = design["platform"]["members"][2]
    # mem_list.append(Member(dict, 1, heading=60))
    # mem_list.append(Member(dict, 1, heading=180))
    # mem_list.append(Member(dict, 1, heading=300))
    # dict = design["platform"]["members"][3]
    # mem_list.append(Member(dict, 1, heading=60))
    # mem_list.append(Member(dict, 1, heading=180))
    # mem_list.append(Member(dict, 1, heading=300))
    
for mem in mem_list:
    mem.setPosition()

for mem in attach_list:
    mem.setPosition()


# dict = design["turbine"]["tower"]
# tower = Member(dict, 1)


from meshmagick.mesh import Mesh, Plane
from meshmagick.mesh_clipper import MeshClipper
from meshmagick.mmio import write_NEM, write_MAR, write_PNL

# mesh = platformMesh(mem_list, sizeMax=1.8, constraint=0.25, recombined=True, clip=False)
# mesh.show()
# print(mesh.nb_vertices, mesh.nb_faces)

# mesh = sPlatformMesh(mem_list[:-1], [mem_list[-1]], sizeMax=1.0, constraint=0.25, recombined=True, clip=False)
# mesh.show()
# print(mesh.nb_vertices, mesh.nb_faces)
mem_list.extend(attach_list)
mesh = platformMesh(mem_list, sizeMax=2.0, constraint=0.25, recombined=True, clip=False)
mesh.show()
print(mesh.nb_vertices, mesh.nb_faces)

from meshmagick.mesh import Mesh

mesh1 = Mesh(vertices=mesh.getVertices(), faces=mesh.getFaces())
mesh1.heal_mesh()


# tower_mesh = platformMesh([tower], sizeMax=1.0, constraint=0.25, recombined=True)
# tower_mesh.show()

write_MAR('models/nemoh/OC4_full.dat', mesh.getVertices(), mesh.getFaces())

# import capytaine as cpt
# from capytaine.post_pro.free_surfaces import FreeSurface

# # hull_mesh = cpt.Mesh(mesh.getVertices(), mesh.getFaces(), 'OC4')
# hull_mesh = cpt.load_mesh('OC4.dat')
# hull_mesh.heal_mesh()

# cal_mesh, lid_mesh = hull_mesh.extract_lid()

# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

# import xarray as xr

# body = cpt.FloatingBody(mesh=cal_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)))

# test_matrix = xr.Dataset(coords={

#     'omega': 0.0628,
#     'wave_direction': np.linspace(0, np.pi/2, 2),
#     'radiating_dof': list(body.dofs),
#     'water_depth': [np.inf],
# })

# dataset = cpt.BEMSolver().fill_dataset(test_matrix, body)

# A11 = dataset.sel(radiating_dof='Surge', influenced_dof='Surge', wave_direction=0)

# a = 1

# dataset.to_dataframe().to_csv('tmp.csv')
# mesh.show()

# print(mesh.nb_vertices, mesh.nb_faces)