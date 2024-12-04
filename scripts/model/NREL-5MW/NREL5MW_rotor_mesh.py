import yaml
from raft.raft_rotor_yang import Rotor, raft_dir
import numpy as np
import os

fname_design = os.path.join(raft_dir, "examples/OC4semi-RAFT_QTF.yaml")

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

meshFile = os.path.join(raft_dir, "models/msh/NREL-5MW-rotor.msh")

rotor.baldeGeo2Mesh(geometry_table, 
                    show=False, 
                    all=True, 
                    mesh=True, 
                    vtk_show=True, 
                    meshFile=meshFile, 
                    shroud_radius=4, 
                    save=True)

