import yaml
from raft.raft_member import Member
from raft.raft_rotor_yang import Rotor

import raft.member2Gmsh as m2g

import matplotlib.pyplot as plt

import numpy as np
import gmsh

from raft.helpers import readCoordinate

import os

# fname_design ="examples/VolturnUS-S_example.yaml"
fname_design ="examples/OC4semi-RAFT_QTF.yaml"

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']

    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0

# dict = design["members"][0]
# mem = Member(dict, 1)

rotor_dict = design["turbine"]
rotor = Rotor(rotor_dict, [1], 0)
rotor.setPosition()

geometry_table = np.array(design["turbine"]['blade'][0]['geometry'])

rotor.baldeGeo2Mesh(geometry_table, show=False, all=True, mesh=True, vtk_show=True, meshFile="../models/msh/NREL-5MW-rotor.msh", shroud_radius=4, save=True)

rotor.bladeGeometry2BladeMember(geometry_table)
# rotor.setYaw()

# rotor.bladeMesh(all=True, mesh=True, show=True)

args = (rotor.ccblade.r, rotor.ccblade.precurve, rotor.ccblade.presweep, rotor.ccblade.precone)

def getCone(r, precurve, presweep, precone):

    n = len(r)

    if np.all(precurve == 0):

        cone = np.ones(n) * precone

    else:

        x_az = -r*np.sin(precone) + precurve*np.cos(precone)
        z_az = r*np.cos(precone) + precurve*np.sin(precone)
        y_az = presweep

        cone = np.zeros(n)

        cone[0] = np.arctan2(-(x_az[1] - x_az[0]), z_az[1] - z_az[0])
        cone[-1] = np.arctan2(-(x_az[-1] - x_az[-2]), z_az[-1] - z_az[-2])
        cone[1:-2] = 0.5*(np.arctan2(-(x_az[1:-2] - x_az[0:-3]), z_az[1:-2] - z_az[0:-3]) + np.arctan2(-(x_az[2:-1]- x_az[1:-2]), z_az[2:-1] - z_az[1:-2]))

    return cone

def rotationMatrix(x3,x2,x1):
    '''Calculates a rotation matrix based on order-z,y,x instrinsic (tait-bryan?) angles, meaning
    they are about the ROTATED axes. (rotation about z-axis would be (0,0,theta) )
    
    Parameters
    ----------
    x3, x2, x1: floats
        The angles that the rotated axes are from the nonrotated axes. Normally roll,pitch,yaw respectively. [rad]

    Returns
    -------
    R : matrix
        The rotation matrix
    '''
    # initialize the sines and cosines
    s1 = np.sin(x1) 
    c1 = np.cos(x1)
    s2 = np.sin(x2) 
    c2 = np.cos(x2)
    s3 = np.sin(x3) 
    c3 = np.cos(x3)
    
    # create the rotation matrix
    R = np.array([[ c1*c2,  c1*s2*s3-c3*s1,  s1*s3+c1*c3*s2],
                  [ c2*s1,  c1*c3+s1*s2*s3,  c3*s1*s2-c1*s3],
                  [   -s2,           c2*s3,           c2*c3]])
    
    return R 

dir = (os.path.dirname(os.path.realpath(__file__)))
gmsh.initialize()

cone = getCone(*args)

curv = []
curvlp = []
# nr = 20
for i,af in enumerate(rotor.ccblade.af):
     
    file = os.path.join(dir, f"designs/airfoil/coordinate/{af.AFName}.txt")

    coord = readCoordinate(file)

    n_p = coord.shape[0]
    r_b_rel = np.array([coord[:, 1], coord[:, 0] - 0.25, np.zeros(n_p)])

    r_b = np.array([-rotor.ccblade.precurve[i], rotor.ccblade.presweep[i], rotor.ccblade.r[i]])

    x = np.zeros(n_p+1)
    y = np.zeros(n_p+1)
    z = np.zeros(n_p+1)

    pts = []
    for j in range (n_p):
        
        x = coord[j, 1] * rotor.ccblade.chord[i] + r_b[0]
        y = coord[j, 0] * rotor.ccblade.chord[i] + r_b[1]
        z = r_b[2]

        r0 = np.array([x,y,z])

        R = rotationMatrix(0.0*np.pi, -cone[i], -rotor.ccblade.theta[i])

        r1 = np.matmul(rotor.R_q, np.matmul(R, r0))
       
        pts.append(gmsh.model.occ.addPoint(r1[0], r1[1], r1[2], rotor.ccblade.chord[i]*0.05))

    pts.append(pts[0])


    curv.append(gmsh.model.occ.addBSpline(pts))
    gmsh.model.occ.addCurveLoop([curv[i]])

###################################################################################
# af = rotor.ccblade.af[1]

# file = os.path.join(dir, f"designs/airfoil/coordinate/{af.AFName}.txt")

# coord = readCoordinate(file)

# n_p = coord.shape[0]
# r_b_rel = np.array([coord[:, 1], coord[:, 0] - 0.25, np.zeros(n_p)])

# r_b = np.array([-cone[1], rotor.ccblade.presweep[1], rotor.ccblade.r[1]])

# x = np.zeros(n_p+1)
# y = np.zeros(n_p+1)
# z = np.zeros(n_p+1)

# for i in range (n_p):
#     x[i] = (coord[i, 1] + r_b[0]) * rotor.ccblade.chord[1]
#     y[i] = (coord[i, 0] + r_b[1]) * rotor.ccblade.chord[1]
#     z[i] = (0.0 + r_b[2]) * rotor.ccblade.chord[1]

# pts = []
# for i in range(n_p):
#         pts.append(gmsh.model.occ.addPoint(x[i], y[i], z[i], 0.5))

# pts.append(pts[0])

# curv.append(gmsh.model.occ.addBSpline(pts))

# ###################################################################################
# af = rotor.ccblade.af[2]

# file = os.path.join(dir, f"designs/airfoil/coordinate/{af.AFName}.txt")

# coord = readCoordinate(file)

# n_p = coord.shape[0]
# r_b_rel = np.array([coord[:, 1], coord[:, 0] - 0.25, np.zeros(n_p)])

# r_b = np.array([-cone[2], rotor.ccblade.presweep[2], rotor.ccblade.r[2]])

# x = np.zeros(n_p+1)
# y = np.zeros(n_p+1)
# z = np.zeros(n_p+1)

# for i in range (n_p):
#     x[i] = (coord[i, 1] + r_b[0]) * rotor.ccblade.chord[2]
#     y[i] = (coord[i, 0] + r_b[1]) * rotor.ccblade.chord[2]
#     z[i] = (0.0 + r_b[2]) * rotor.ccblade.chord[2]

# pts = []
# for i in range(n_p):
#         pts.append(gmsh.model.occ.addPoint(x[i], y[i], z[i], 0.5))

# pts.append(pts[0])

# curv.append(gmsh.model.occ.addBSpline(pts))

###################################################################################
# gmsh.model.occ.addCurveLoop([1], 1)
# gmsh.model.occ.addCurveLoop([2], 2)
# gmsh.model.occ.addCurveLoop([3], 3)

for iter in range(len(curv) - 1):
    gmsh.model.occ.addThruSections([curv[iter], curv[iter+1]], iter+1, makeSolid="False")

# gmsh.fltk.run()
# flag = len(curv)
# for i in range (2, len(curv) - 1):
#     if i == 2:
#         gmsh.model.occ.fuse([(3,1)], [(3, i)], flag)
#     else:
#         gmsh.model.occ.fuse([(3,flag)], [(3, i)], flag+1)
#         flag += 1
gmsh.model.occ.fuse([(3,1)], [(3, i) for i in range(2, len(curv))])
# gmsh.model.occ.fuse
# gmsh.option.setNumber("Geometry.OCCThruSectionsDegree", 2)
# gmsh.model.occ.addThruSections(curv, 1, continuity="C0", parametrization="Centripetal")
# gmsh.model.occ.addThruSections(curv[1:3], 2)
# gmsh.model.occ.addThruSections(curv[2:4], 3)
# gmsh.model.occ.addThruSections(curv[3:5], 4)
gmsh.model.occ.synchronize()

gmsh.fltk.run()


# def memberMeshBlade(blade_r, af):





rotor.bladeGeometry2Member()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
rotor.bladeMemberList[0].setPosition()
rotor.bladeMemberList[0].plotSurface(ax)
plt.show()
a = 1

# mem.setPosition()

# if mem.shape == "rectangular":
#     m2g.memberMeshRect(mem.stations, mem.sl, mem.R, index=0)
# elif mem.shape == "circular":
#      m2g.memberMeshCirc(mem.stations, mem.d, mem.rA, mem.rB, index=0)

# m2g.writeMesh(visual=True, meshFile="mem_subm_horz_rect_cyl.msh", hydroFlag=False)