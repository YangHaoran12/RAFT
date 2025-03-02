# import os 

# import numpy as np
# dir = (os.path.dirname(os.path.realpath(__file__)))

# # a = np.loadtxt(os.path.join(dir, "designs/airfoil/polar/Cylinder1.dat"))

# a = np.loadtxt(os.path.join(dir, "designs/airfoil/coordinate/Cylinder1.txt"), skiprows=8)

# a = 1

# print(len([a]))

# import yaml
# import numpy as np

# fname_design ="designs/tmp.yaml"

# with open(fname_design) as file:
#     design = yaml.load(file, Loader=yaml.FullLoader)

#     airfoil = design["airfoils"]

# x = np.array(airfoil[7]["coordinates"]['x'])
# y = np.array(airfoil[7]["coordinates"]['y'])

# # x.reshape(len(x), 1)
# # y.reshape(len(y), 1)


# tmp = np.stack((x,y), axis=-1)


# np.savetxt("designs/airfoil/coordinate/FFA-W3-360.txt", tmp, '%10.6f')

# print(x.shape[0])

# a = 1

import numpy as np
import pyhams.pyhams as ph

addedMass, damping, w = ph.read_wamit1('tmp/Buoy.1')

A11 = addedMass[0,0,:]
B11 = damping[0,0,:]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2)

w /= (2*np.pi)

ax[0,0].plot(w, A11)

ax[0,1].plot(w, B11)

ax[1,0].plot(w, A11*1025)
ax[1,1].plot(w, B11*w*1025)

plt.show()



mod, phase, real, imag, w, headings = ph.read_wamit3('tmp/Buoy.3')

w /= (2*np.pi)

fig, ax = plt.subplots(2,1)
ax[0].plot(w/np.pi/2, mod[0,0,:])
ax[1].plot(w, 9.81*1025*mod[0,0,:])

plt.show()


a = 1