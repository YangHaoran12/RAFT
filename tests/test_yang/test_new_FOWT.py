from raft.raft_member import Member, BladeMember, TowerMember
from raft.raft_mesh import platformMesh, sPlatformMesh
import yaml
import numpy as np
import moorpy as mp

from raft.raft_fowt import FOWT
from raft.raft_fowt_orig import FOWT as FOWT_old
from raft.raft_model import Model
from raft.helpers import JONSWAP

fname_design ="designs/VolturnUS-S.yaml"
# fname_design ="designs/OC4semi.yaml"

    # fname_design = f"tests/test_data/{list_files[0]}"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# model = Model(design, 1)
# model.analyzeUnloaded()
# model.analyzeCases(RAO_plot=True)
import matplotlib.pyplot as plt

# ax = plt.figure().add_subplot(projection='3d')
# model.plot(ax)
# plt.show()

min_freq = 0.05 * np.pi*2
max_freq = 0.2 * np.pi*2
fowt = FOWT(design, np.linspace(min_freq, max_freq, 81), None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)
case = dict(zip(design['cases']['keys'], design['cases']['data'][0]))
case1 = dict(zip(design['cases']['keys'], design['cases']['data'][0]))
fowt.setPosition([0,0,0,0,0,0])
fowt.calcStatics()

# V_required = (fowt.V*fowt.rho_water - fowt.m_shell) / fowt.rho_water
# V_ballast  = np.sum([mem.V for mem in fowt.memberList])

# fowt.calcTowerAeroLoads(case)
fowt.solveStatics(case)
fowt.solveDynamics(case, RAO_plot=True)

plt.show()

results = {}
fowt.saveTurbineOutputs(results=results, case=case)

results1 = {}
fowt.solveStatics(case1)
fowt.solveDynamics(case1, RAO_plot=False)
fowt.saveTurbineOutputs(results=results1, case=case1)

import os
from raft.raft_rotor import raft_dir
import pandas as pd


fast_results = pd.read_csv(os.path.join(raft_dir, 'scripts/results.csv'))

fig, ax = plt.subplots(6, 2, sharex=True, figsize=(6,8))
TwoPi = np.pi*2
# loop through each FOWT and plot its response (on the same figure for now)
metrics = results
ax[0,0].plot(fowt.w / TwoPi, TwoPi * metrics['surge_PSD'][:], '--r', label='RAFT')  # surge
ax[0,0].plot(fast_results['f'], fast_results['psd_Surge'], label='OpenFAST')
ax[1,0].plot(fowt.w / TwoPi, TwoPi * metrics['heave_PSD'][:], '--r')  # heave
ax[1,0].plot(fast_results['f'], fast_results['psd_Heave'])
ax[2,0].plot(fowt.w / TwoPi, TwoPi * metrics['pitch_PSD'][:], '--r')
ax[2,0].plot(fast_results['f'], fast_results['psd_Pitch'])
ax[3,0].plot(fowt.w / TwoPi, TwoPi * metrics['Tmoor_PSD'][3,:], '--r')
ax[3,0].plot(fast_results['f'], fast_results['psd_tension'])
ax[4,0].plot(fowt.w / TwoPi, TwoPi * metrics['AxRNA_PSD'][:], '--r')
ax[4,0].plot(fast_results['f'], fast_results['psd_NacAx'])
ax[5,0].plot(fowt.w / TwoPi, TwoPi * metrics['Mybase_PSD'][:]/1e6, '--r')
ax[5,0].plot(fast_results['f'], fast_results['psd_Bmy'])

metrics = results1
fast_results = pd.read_csv(os.path.join(raft_dir, 'scripts/results1.csv'))
ax[0,1].plot(fowt.w / TwoPi, TwoPi * metrics['surge_PSD'][:], '--r', label='RAFT')  # surge
ax[0,1].plot(fast_results['f'], fast_results['psd_Surge'], label='OpenFAST')
ax[1,1].plot(fowt.w / TwoPi, TwoPi * metrics['heave_PSD'][:], '--r')  # heave
ax[1,1].plot(fast_results['f'], fast_results['psd_Heave'])
ax[2,1].plot(fowt.w / TwoPi, TwoPi * metrics['pitch_PSD'][:], '--r')
ax[2,1].plot(fast_results['f'], fast_results['psd_Pitch'])
ax[3,1].plot(fowt.w / TwoPi, TwoPi * metrics['Tmoor_PSD'][3,:], '--r')
ax[3,1].plot(fast_results['f'], fast_results['psd_tension'])
ax[4,1].plot(fowt.w / TwoPi, TwoPi * metrics['AxRNA_PSD'][:], '--r')
ax[4,1].plot(fast_results['f'], fast_results['psd_NacAx'])
ax[5,1].plot(fowt.w / TwoPi, TwoPi * metrics['Mybase_PSD'][:]/1e6, '--r')
ax[5,1].plot(fast_results['f'], fast_results['psd_Bmy'])

ax[0,0].set_ylabel('surge \n'+r'(m$^2$/Hz)')
ax[1,0].set_ylabel('heave \n'+r'(m$^2$/Hz)')
ax[2,0].set_ylabel('pitch \n'+r'(deg$^2$/Hz)')
ax[3,0].set_ylabel('moor. ten. \n'+r'((m/s$^2$)$^2$/Hz)')
ax[4,0].set_ylabel('Nac. Ax \n'+r'((kN·m)$^2$/Hz)')
ax[5,0].set_ylabel('twr. bend.\n'+r'(m$^2$/Hz)')

ax[-1,0].set_xlabel('frequency (Hz)')  
ax[-1,1].set_xlabel('frequency (Hz)')
ax[0,0].legend()
fig.suptitle('RAFT power spectral densities')
for ax_ in ax:
    ax_[0].grid(True)
    ax_[1].grid(True)
# ax[0].grid(True)
plt.subplots_adjust(left=0.2)
plt.xlim(0.05, 0.2)
plt.show()



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