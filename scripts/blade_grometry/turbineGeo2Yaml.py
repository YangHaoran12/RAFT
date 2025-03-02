import numpy as np
import yaml
from scipy.interpolate      import PchipInterpolator


geo = np.loadtxt('NRELOffshrBsline5MW_AeroDyn_blade.dat', skiprows=6, max_rows=19, usecols=[0,1,2,4,5])
rHub = 1.5
rTip = 62.99
r = geo[:,0]+rHub
chord = geo[:,-1]
theta = geo[:, -2]
precurve = geo[:, 1]
presweep = geo[:, 2]

pitch = np.loadtxt('NRELOffshrBsline5MW_Blade.dat', skiprows=16, max_rows=49, usecols=[0,1])

spline = PchipInterpolator(pitch[:,0]*61.5, pitch[:,1])
pitch_interp = spline(r)
width = 8


fname_design = '../../examples/OC4semi-RAFT_QTF.yaml'

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["turbine"]['blade']['geometry']

r_ = np.linspace(rHub, rTip, 31)

chord_ = np.interp(r_, r, chord)
theta_ = np.interp(r_, r, theta)
precurve_ = np.interp(r_, r, precurve)
presweep_ = np.interp(r_, r, presweep)
pitch_ = np.interp(r_, r, pitch_interp)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2)
ax[0][0].plot(r_, chord_)
ax[0][1].plot(r_, presweep_)
ax[1][0].plot(r_, theta_)
ax[1][1].plot(r_, precurve_)

plt.show()


geometry = np.column_stack((r_,chord_, theta_, precurve_, presweep_, pitch_))

np.savetxt('NREL5MW_blade_geometry.txt', geometry, '%10.3f')
# import pprint
# pprint.pprint(geometry)


##################################################################################
geo = np.loadtxt('IEA-15-240-RWT_AeroDyn15_blade.dat', skiprows=6, usecols=[0,1,2,4,5])

rHub = 3.97
rTip = 120.969
r = geo[:,0]+rHub
chord = geo[:,-1]
theta = geo[:, -2]
precurve = geo[:, 1]
presweep = geo[:, 2]

pitch = np.loadtxt('IEA-15-240-RWT_ElastoDyn_blade.dat', skiprows=16, max_rows=50, usecols=[0,1])

spline = PchipInterpolator(pitch[:,0]*121, pitch[:,1])
pitch_interp = spline(r)
width = 8

fname_design = '../../examples/VolturnUS-S_example.yaml'

with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

dict = design["turbine"]['blade']['geometry']

r_ = np.linspace(rHub, rTip, 51)

chord_ = np.interp(r_, r, chord)
theta_ = np.interp(r_, r, theta)
precurve_ = np.interp(r_, r, precurve)
presweep_ = np.interp(r_, r, presweep)
pitch_ = np.interp(r_, r, pitch_interp)

def getCone(r, precurve, precone):
    n = len(r)

    if np.all(precurve == 0):      
        cone = np.ones(n) * precone

    else:
        x_az = -r*np.sin(precone) + precurve*np.cos(precone)
        z_az = r*np.cos(precone) + precurve*np.sin(precone)
        # y_az = presweep

        cone = np.zeros(n)

        cone[0] = np.arctan2(-(x_az[1] - x_az[0]), z_az[1] - z_az[0])
        cone[-1] = np.arctan2(-(x_az[-1] - x_az[-2]), z_az[-1] - z_az[-2])
        cone[1:-1] = 0.5*(np.arctan2(-(x_az[1:-1] - x_az[0:-2]), z_az[1:-1] - z_az[0:-2]) + np.arctan2(-(x_az[2:]- x_az[1:-1]), z_az[2:] - z_az[1:-1]))

    return cone

cone = getCone(r_, precurve_, np.deg2rad(4.0))


fig, ax = plt.subplots(2,2)
ax[0][0].plot(r_, chord_)
ax[0][1].plot(r_, cone)
ax[1][0].plot(r_, theta_)
ax[1][1].plot(r_, precurve_)

plt.show()


# spline = PchipInterpolator(r, chord)
# chord_ = spline(r_)
# spline = PchipInterpolator(r, theta)
# theta_ = spline(r_)
# spline = PchipInterpolator(r, precurve)
# precurve_ = spline(r_)
# spline = PchipInterpolator(r, presweep)
# presweep_ = spline(r_)

# fig1, ax1 = plt.subplots(2,2)
# ax1[0][0].plot(r_, chord_)
# ax1[0][1].plot(r_, presweep_)
# ax1[1][0].plot(r_, theta_)
# ax1[1][1].plot(r_, precurve_)

# plt.show()



geometry = np.column_stack((r_, chord_, theta_, precurve_, presweep_, pitch_))

np.savetxt('IEA-15-240-RWT.txt', geometry, '%10.3f')

# def format_aligned_array(array):
#     return [[f'{x:>{width}.3f}' for x in row] for row in array]



# aligned_array = format_aligned_array(geometry)
# import pprint
# pprint.pprint(aligned_array)

# 自定义 YAML 转储器，确保每一行都是流式格式
# def custom_representer(dumper, value):
#     return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)

# # 注册自定义的代表器
# yaml.add_representer(list, custom_representer)

# 将对齐后的数组转储到 YAML 文件
# with open('NREL5MW_blade_geometry.yaml', 'w') as file:
#     yaml.dump(aligned_array, file, default_flow_style=True, indent=2)
# with open('NREL5MW_blade_geometry.yaml', 'w') as f:
#     yaml.dump(geometry.tolist(), f, default_flow_style=True, indent=2, width=80)

# NREL5MW

# IEA15MW = {}
# np.loadtxt('IEA-15-240-RWT_AeroDyn15_blade.dat')