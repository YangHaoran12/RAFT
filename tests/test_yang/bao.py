from raft.raft_member import Member
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np
import moorpy as mp

from raft.raft_fowt import FOWT

fname_design ="examples/bao.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)


moor = mp.System()
moor.parseYAML(design['mooring'])

# ensure proper setup with one coupled Body tied to this FOWT
if len(moor.bodyList) == 0:
    moor.addBody(-1, [0,0,0,0,0,0]) # create a new body if needed
    for point in moor.pointList:
        if point.type == -1:  # attached any coupled points to the body
            moor.bodyList[0].attachPoint(point.number, point.r)
            point.type = 1  # now indicate point is fixed (to the body)
            
elif len(moor.bodyList) == 1:
    moor.bodyList[0].type = -1  # ensure it's set to coupled type
else:
    raise Exception("More than one body detected in FOWT mooring system.")
    
# move mooring system according to the FOWT's reference position

moor.initialize()

# moor.initialize()

moor.bodyList[0].setPosition([0,0,0,0,0,0])
moor.solveEquilibrium()                                       # equilibrate
# fig, ax = moor.plot()  

import matplotlib.pyplot as plt

C = moor.getCoupledStiffnessA()
# C_ = moor.getCoupledStiffness()
print(C)

moor.bodyList[0].setPosition([5,0,0,np.pi/8,0,0])
moor.solveEquilibrium()

fig, ax = moor.plot(color='red') 

# fig,ax = plt.subplots(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
fowt = FOWT(design, [0.628, 0.628*2], None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)

# fowt.tmp_plot(ax, plot_rotor=False, plot_ms=False)

# mesh = platformMesh(fowt.memberList[:10], clip=False)
# mesh.save()

fowt.calcStatics()

fowt.hydroPath = 'tmp/Platform'

import pyhams.pyhams as ph

addedMass, damping, w = ph.read_wamit1('tmp/Buoy.1')

fowt.readHydro()

plt.show()

a = 1