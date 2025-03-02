import yaml
from raft.raft_rotor import Rotor, raft_dir
import numpy as np
import os

fname_design = os.path.join(raft_dir, "examples/VolturnUS-S_example_copy.yaml")
# fname_design ="examples/OC4semi-RAFT_QTF.yaml"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']

    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0


rotor_dict = design["turbine"]
rotor = Rotor(rotor_dict, [1], 0)
rotor.setPosition()

geometry_table = np.array(design["turbine"]['blade'][0]['geometry'])

meshFile = os.path.join(raft_dir, "models/msh/IEA-15MW-rotor.msh")

# geometry_table[:, 3] = -geometry_table[:, 3]

rotor.bladeGeo2Mesh(geometry_table, 
                    pitch=2.82,
                    show=True, 
                    all=True, 
                    mesh=False, 
                    vtk_show=False, 
                    meshFile=meshFile, 
                    mesh_size_max=0.2,
                    shroud_radius=5.0, 
                    save=False,
                    constraint=0.025)

meshDir = os.path.join(raft_dir, "temp")
rotor.bladdeGeo2AbsPts(geometry_table, pitch=0, meshDir=meshDir)

a = 1