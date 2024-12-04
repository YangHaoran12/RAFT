from raft.raft_member_yang import Member, BladeMember, TowerMember
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np
import moorpy as mp

from raft.raft_fowt_yang import FOWT
from raft.raft_fowt import FOWT as FOWT_old
from raft.raft_model import Model
from raft.helpers import JONSWAP

# fname_design ="designs/VolturnUS-S.yaml"
fname_design ="designs/OC4semi.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)


fowt = FOWT(design, np.linspace(0.0628, np.pi*0.8, 40), None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)
fowt1 = FOWT_old(design, np.linspace(0.0628, np.pi*0.8, 40), None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)



r6 = np.array([1,0,0,np.deg2rad(0),np.deg2rad(1),np.deg2rad(0)])


fowt.setPosition(r6)
fowt.calcStatics()
from tabulate import tabulate

print(tabulate(fowt.C_hydro))
print(fowt.W_hydro)


fowt1.setPosition(r6)
fowt1.calcStatics()
print(tabulate(fowt1.C_hydro))
print(fowt1.W_hydro)

a = 1
# model = Model(design, 1)
# # fowt.tmp_plot(ax, plot_rotor=False, plot_ms=False)

# # mesh = platformMesh(fowt.memberList[:10], clip=False)
# # mesh.save()
# import matplotlib.pyplot as plt
# model.analyzeUnloaded()
# model.analyzeCases(RAO_plot=1)

# S = JONSWAP(ws=model.w, Tp=13.4, Hs=8.5)
# w = model.w
# dw = model.w[1] - model.w[0]

# eta = np.sqrt(2*S*dw)

# results = {}

# results['omega'] = w
# results['eta'] = eta
# results['RAO_Surge'] = np.abs(model.Xi[0,0,:])
# results['RAO_Sway'] =  np.abs(model.Xi[0,1,:])
# results['RAO_Heave'] = np.abs(model.Xi[0,2,:])
# results['RAO_Roll']  = np.abs(model.Xi[0,3,:])
# results['RAO_Pitch'] = np.abs(model.Xi[0,4,:])
# results['RAO_Yaw']   = np.abs(model.Xi[0,5,:])

# import pandas as pd

# pd.DataFrame(results).to_csv('RAO.csv')

# plt.show()
fowt.setPosition(np.zeros(6))
# fowt1.setPosition(np.zeros(6))

fowt.calcStatics()
fowt.calcHydroConstants()
fowt.solveEigen(display=1)


model.analyzeUnloaded()
model.solveEigen(display=1)

fowt.calcStatics()
fowt1.calcStatics()

fowt.calcHydroConstants()
fowt1.calcHydroConstants()

# fowt.readHydro()

fowt.calcBEM(singleBody=True, 
             sizeMax=1.8, 
             nw=101, 
             headings=np.linspace(0,180,7, endpoint=True), 
             meshDir='VolturnUS-S',
             saveMesh=True
            )

fowt.solveEigen(display=1)
fowt1.solveEigen(display=1)

B = fowt.B_BEM

b11 = B[0,0,:]
b22 = B[1,1,:]
b33 = B[2,2,:]

import matplotlib.pyplot as plt
plt.plot(np.linspace(0.0628, np.pi*0.8, 40), b11)
plt.plot(np.linspace(0.0628, np.pi*0.8, 40), b22)
plt.plot(np.linspace(0.0628, np.pi*0.8, 40), b33)

plt.show()

import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d')

fowt.plot(ax)
plt.show()

fowt1 = FOWT_old(design, [0.628, 0.628*2], None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)
fowt1.calcStatics()

new = vars(fowt)['props']
old = vars(fowt1)['props']


import pandas as pd

pd.DataFrame(new).to_csv('new.csv')
pd.DataFrame(old).to_csv('old.csv')

fowt.hydroPath = 'tmp/Platform'

import pyhams.pyhams as ph

addedMass, damping, w = ph.read_wamit1('tmp/Buoy.1')

fowt.readHydro()

plt.show()

a = 1