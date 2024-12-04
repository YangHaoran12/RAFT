from raft.raft_member_yang import Member, TowerMember
from raft.raft_mesh import platformMesh
from raft.raft_fowt_yang import FOWT
from raft.raft_rotor_yang import raft_dir
import yaml
import numpy as np
import os

fname_design ="designs/OC4semi.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)
    
    fowt = FOWT(design, [0.1,0.2], None)
    fowt.setPosition(np.zeros(6))

# fowt.calcStatics()

from meshmagick.mesh import Mesh, Plane
from meshmagick.mmio import write_NEM, write_MAR, write_PNL,write_STL, write_VTK

mesh = platformMesh(fowt.memberList[:-1], 
                    sizeMax=2.0, 
                    constraint=0.33, 
                    recombined=True, 
                    clip=False)
mesh.show()
mesh.save('tmp.vtp')
print(mesh.nb_vertices, mesh.nb_faces)

from meshmagick.mesh import Mesh

mesh1 = Mesh(vertices=mesh.getVertices(), faces=mesh.getFaces())
mesh1.heal_mesh()
mesh1.show()

# tower_mesh = platformMesh([tower], sizeMax=1.0, constraint=0.25, recombined=True)
# tower_mesh.show()

meshFile = os.path.join(raft_dir, f'models/nemoh/OC4_full.dat')
meshFile_stl = os.path.join(raft_dir, f'models/nemoh/OC4_full.stl')
meshFile_vtk= os.path.join(raft_dir, f'models/nemoh/OC4_full.vtk')

# write_MAR(meshFile, mesh1.vertices, mesh1.faces)
write_STL(meshFile_stl, mesh1.vertices, mesh1.faces)
write_VTK(meshFile_vtk, mesh1.vertices, mesh1.faces)
