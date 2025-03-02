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

fowt_ = fowt.copy()

from raft.opt_helpers import calcStaticTwrBendingMoment

M, stress = calcStaticTwrBendingMoment(fowt_, case)

a = 1