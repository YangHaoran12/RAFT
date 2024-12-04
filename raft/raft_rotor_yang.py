# RAFT's rotor class

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from raft.pyIECWind         import pyIECWind_extreme
from raft.raft_member_yang import Member, BladeMember

from scipy.interpolate      import PchipInterpolator
from scipy.special          import modstruve, iv

from raft.airfoilprep import Polar

from raft.helpers import rotationMatrix, getFromDict, rotateMatrix3, rotateMatrix6, RotFrm2Vect, readPolar, readCoordinate

try:
    from ccblade.ccblade import CCBlade, CCAirfoil
except:
    from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

from typing import Iterable
import pickle

if False:
    thrust_psd = pickle.load( open( "/Users/dzalkind/Tools/RAFT/designs/rotors/thrust_psd.p", "rb" ) )


# global constants
raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
rad2deg = 57.2958
rpm2radps = 0.1047

# a class for the rotor structure, aerodynamics, and control in RAFT
class Rotor:

    def __init__(self, turbine, w, ir):
        '''
        >>>> add mean offset parameters add move this to runCCBlade<<<<
        ir = the index of the values in the arrays of the input file for multiple rotors
        '''

        self.w = np.array(w)
        self.nw = len(self.w)
        self.turbine = turbine      # store dictionary for later use
        
        # self.r_rel is the position of the RNA reference point (which the RNA yaws about) on the FOWT
        if 'rRNA' in turbine: # Temporary if statement. Need to fix getFromDict to handle this
            self.r_rel = getFromDict(turbine, 'rRNA', shape=[turbine['nrotors'], 3])[ir]
        else:
            if turbine['nrotors'] > 1:
                raise Exception("For designs with more than one rotor, the RNA reference point must be specified for each of them.")
            self.r_rel = [0, 0, 100.]

        self.overhang   = getFromDict(turbine, 'overhang', shape=turbine['nrotors'])[ir]  # rotor offset in +x before yaw [m]
        if self.overhang > 0:
            print("WARNING: The turbine overhang input was positive for upwind turbines.")
            print("The sign convention is now along +x (opposite of before)")
            print("so that the specified overhang is a downwind rotor. Please flip the")
            print("sign (negative overhang) if you wanted an upwind rotor.")
            
        self.xCG_RNA = getFromDict(turbine, 'xCG_RNA', shape=turbine['nrotors'])[ir]  # RNA CG offset in +x before yaw [m]
        
        # mass/inertia
        self.mRNA    = getFromDict(turbine, 'mRNA'   , shape=turbine['nrotors'])[ir]
        self.IxRNA   = getFromDict(turbine, 'IxRNA'  , shape=turbine['nrotors'])[ir]
        self.IrRNA   = getFromDict(turbine, 'IrRNA'  , shape=turbine['nrotors'])[ir]
        
        self.speed_gain = getFromDict(turbine, 'speed_gain', shape=turbine['nrotors'], default=1.0)[ir]
        
        self.nBlades    = getFromDict(turbine, 'nBlades', shape=turbine['nrotors'], dtype=int)[ir]         # [-]
        
        self.platform_heading = 0  # stored copy of the yaw of the FOWT [rad]
        self.yaw = 0   # the relative yaw rotation of this rotor, as from a yaw drive [rad]

        # Set reference headings for yaw control that may vary by load case
        self.inflow_heading = 0  # global heading of the inflow vector [rad]
        self.turbine_heading = 0  # global turbine heading (excluding toe) [rad]

        # how nacelle yaw is handled: 0=assume aligned, 1=use case info, 2/3=use self.yaw_command value
        self.yaw_mode = getFromDict(turbine, 'yaw_mode', shape=turbine['nrotors'], dtype=int, default=0)[ir]
        self.yaw_command = 0  # f yaw_mode==1: yaw misalignment; 2: relative to platform; 3: absolute heading [rad]
        
        # Blade reference azimuth angles (may be unnecessary)
        default_azimuths    = list(np.arange(self.nBlades) * 360. / self.nBlades) # equally distribute blades
        self.azimuths       = getFromDict(turbine, 'headings', shape=-1, default=default_azimuths)  # [deg]
        
        self.Rhub       = getFromDict(turbine, 'Rhub', shape=turbine['nrotors'])[ir]            # [m]
        self.precone    = getFromDict(turbine, 'precone', shape=turbine['nrotors'])[ir]         # [m]
        
        # Rotor shaft tilt and toe angles in directions of pitch and yaw, respectively [rad]
        self.shaft_tilt = getFromDict(turbine, 'shaft_tilt', shape=turbine['nrotors'])[ir]*np.pi/180
        self.shaft_toe  = getFromDict(turbine, 'shaft_toe', shape=turbine['nrotors'],
                                      default=0)[ir]*np.pi/180
        
        self.aeroServoMod = getFromDict(turbine, 'aeroServoMod', shape=turbine['nrotors'], default=1)[ir]  # flag for aeroservodynamics (0=none, 1=aero only, 2=aero and control)

        # Unit vector of rotor axis, facing downflow [-]. Relative to the FOWT. Includes shaft tilt and initial yaw
        self.q_rel = np.matmul(rotationMatrix(0, self.shaft_tilt, self.shaft_toe), np.array([1.,0.,0.]) )        
        
        # initialize absolute position/orientation variables
        self.r3 = np.zeros(3)  # instantaneous global position of rotor hub location       
        self.q  = np.array(self.q_rel)    # instantaneous unit vector of rotor axis

        self.R_ptfm  = np.ones(3)  # rotation matrix for platform orientation 
        
        # If `hHub` is specified, we overwrite the z-coordinate of the RNA reference point
        if 'hHub' in turbine:
            hHub    = getFromDict(turbine, 'hHub'   , shape=turbine['nrotors'])[ir]  # overwrites r_rel[2] [m]
            self.r_rel[2] = hHub - self.q[2]*self.overhang
        self.hHub = self.r_rel[2] + self.q[2]*self.overhang  # we may not need this anymore
        self.Zhub = self.hHub

        # below is initialization, needs to be updated by setPosition...
        self.r_RRP = np.array(self.r_rel)  # RNA reference point
        self.r_CG  = np.array(self.r_rel)  # RNA CG location
        self.r_hub = np.array(self.r_rel)  # rotor hub coordinates in global [m]

        # Call setPosition to properly initialize various location parameters
        self.setPosition()

        # support if blade and wt_ops are each a single dictionary or a list of dictionaries for each rotor
        # note: this would only be done on the first turbine  >>> we may want to move this stuff up a level, outside of each Rotor object
        if isinstance(turbine['blade'], dict):          # if we use the entire blade dict list approach
            turbine['blade'] = [turbine['blade']]*turbine['nrotors']
        if isinstance(turbine['wt_ops'], dict):
            turbine['wt_ops'] = [turbine['wt_ops']]*turbine['nrotors']

        self.R_rot      = getFromDict(turbine['blade'][ir], 'Rtip', shape=-1)
        # otherwise, we can avoid the if statment if we use the list approach within each dict value of the blade dict
        
        
        # Ensure the blade geometry inputs are inputted correctly
        nb = len(turbine['blade'])      # the number of blade types in the system.
            # Reminder: Assumption is that all blades on a rotor will be the same. You can input nrotors blade types for multiple rotors, or input one blade type and have it be used for all rotors in the system
        for ib in range(nb):
            nr = len(turbine['blade'][0]['geometry'])       # the number of nodes along the length of the blade
            r0 = turbine['blade'][ib]['geometry'][0][0]     # the first radius value of the blade
            rtip = turbine['blade'][ib]['geometry'][-1][0]  # the last radius value of the blade
            if r0 >= self.Rhub and rtip <= self.R_rot:  # if the geometry range lies between the hub radius and the blade length, do nothing, this is normal
                pass   
            #elif rtip-r0 == self.R_rot-self.Rhub and r0 < self.Rhub:     # if the geometry range is the right length, but starts inside the hub radius [0,length-Rhub], shift values to outside the hub [Rhub,length] 
                #print("WARNING: The input blade geometry radii start inside the hub, when they need to be defined outside of the hub")
                #for i in range(nr):
                    #turbine['blade'][ib]['geometry'][i][0] += self.Rhub-r0
            elif r0 < self.Rhub or rtip > self.R_rot:           # if none of the above works, then something is off with the input blade radii
                raise ValueError(f"Input blade geometry is invalid. First node radius needs to be >= Rhub ({self.Rhub}) or last node radius needs to be <= Rtip ({self.R_rot})")


        self.Uhub      = getFromDict(turbine['wt_ops'][ir], 'v', shape=-1)
        self.Omega_rpm = getFromDict(turbine['wt_ops'][ir], 'omega_op', shape=-1)
        self.pitch_deg = getFromDict(turbine['wt_ops'][ir], 'pitch_op', shape=-1)

        self.I_drivetrain = getFromDict(turbine, 'I_drivetrain', shape=turbine['nrotors'])[ir]

		# Add parked pitch, rotor speed, assuming fully shut down by 40% above cut-out
        self.Uhub = np.r_[self.Uhub, self.Uhub.max()*1.4, 100]
        self.Omega_rpm = np.r_[self.Omega_rpm, 0, 0]
        self.pitch_deg = np.r_[self.pitch_deg, 90, 90]
		
        # Set default control gains
        self.kp_0 = np.zeros_like(self.Uhub)
        self.ki_0 = np.zeros_like(self.Uhub)       
        self.k_float = 0 # np.zeros_like(self.Uhub), right now this is a single value, but this may change    <<< what is this?

        # arrays to hold water wave velocities and accelerations if applicable
        self.u  = np.array([[[]]])
        self.ud = np.array([[[]]])
        
        self.f0 = np.zeros(6)  # mean forces and moments about hub (aligned with platform local directions)

        # Set CCBlade flags
        tiploss = True # Tip loss model True/False
        hubloss = True # Hub loss model, True/False
        wakerotation = True # Wake rotation, True/False
        usecd = True # Use drag coefficient within BEMT, True/False


        # ----- AIRFOIL STUFF ------
        
        self.AFMode = getFromDict(turbine, 'AFMode', shape=0, default=0)
        # compile info for airfoil station points along the blade
        #nStations = len(turbine['blade'][ir]["airfoils"])
        station_airfoil  = [ b for [a,b] in turbine['blade'][ir]["airfoils"] ]  # airfoil name
        station_position = [ a for [a,b] in turbine['blade'][ir]["airfoils"] ]  # airfoil relative position from blade root to tip [0-1]
        nStations = len(station_airfoil)
        
        # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
        n_aoa = 200 # [-] - number of angles of attack to discretize airfoil polars - MUST BE MULTIPLE OF 4
        aoa = np.unique(np.hstack([np.linspace(-180, -30, int(n_aoa/4.0 + 1)), 
                                   np.linspace( -30,  30, int(n_aoa/2.0),),
                                   np.linspace(  30, 180, int(n_aoa/4.0 + 1))]))

        # compile info for individual airfoils
        n_af = len(turbine["airfoils"])  #len(np.unique(station_airfoil))  # number of airfoils that are used in the rotor
        airfoil_name = n_af * [""]         # name of each listed airfoil
        airfoil_Re   = n_af * [0]
        airfoil_thickness = np.zeros(n_af) # relative thickness of each listed airfoil (thickness/chord)
        Ca = np.zeros([n_af, 2])           # added mass coefficient [edgewise, flapwise] of each airfoil
        
        for i in range(n_af):
            airfoil_name[i] = turbine["airfoils"][i]["name"]
            airfoil_thickness[i] = getFromDict(turbine["airfoils"][i], "relative_thickness", shape=0, default=0.2)
            airfoil_Re[i] = getFromDict(turbine['airfoils'][i], 'Re', shape=0, default=0.0)
            Ca[i,:] = getFromDict(turbine["airfoils"][i], "added_mass_coeff", shape=-1, default=[0.5, 1.0])
            # if 'added_mass_coeff' in turbine["airfoils"][i].keys():
            #     Ca[i,:] = turbine["airfoils"][i]["added_mass_coeff"]
            # else:
            #     Ca[i,:] = [0.5, 1.0]  # default added mass coefficients if not supplied

        # cl = np.zeros((n_af, n_aoa, 1))
        # cd = np.zeros((n_af, n_aoa, 1))
        # cm = np.zeros((n_af, n_aoa, 1))
        cpmin = np.zeros((n_af, n_aoa, 1))
        if len(np.array(turbine["airfoils"][i]['data'])[0]) > 4:
            cpmin_flag = True
        else:
            cpmin_flag = False

        polar_list = []

        # Interp cl-cd-cm along predefined grid of angle of attack
        for i in range(n_af):

            if self.AFMode == 0:         
                polar_table = np.array(turbine["airfoils"][i]['data'])
                polar = Polar(airfoil_Re[i], polar_table[:,0], polar_table[:,1], polar_table[:,2], polar_table[:,3])

            elif self.AFMode == 1:
                polar_file = os.path.join(raft_dir, "designs/airfoil/polar", f"{airfoil_name[i]}.dat")
                polar_table = readPolar(polar_file)
                polar = Polar(airfoil_Re[i], polar_table[:,0], polar_table[:,1], polar_table[:,2], polar_table[:,3])
                

            elif self.AFMode == 2:

                pass

                # if abs(polar_table[0,0] + 180) > 1e-5 or abs(polar_table[-1,0] - 180) <1e-5:
                #     polar.extrapolate(cdmax=1.5, nalpha=15)
            else:

                raise ValueError("AFMode should be 0, 1 or 2")
            
            # Note: polar_table[:,0] must be in degrees
            # cl[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,1])
            # cd[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,2])
            # cm[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,3])
            if cpmin_flag:
                cpmin[i, :, 0] = np.interp(aoa, polar_table[:,0], polar_table[:,4])
                        
            # if abs(polar.cl[i, 0, 0] - polar.cl[i, -1, 0]) > 1.0e-5:
            #     print("WARNING: Ai " + airfoil_name[i] + " has the lift coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
            #     polar.cl[i, 0, 0] = polar.cl[i, -1, 0]
            # if abs(polar.cd[i, 0, 0] - polar.cd[i, -1, 0]) > 1.0e-5:
            #     print("WARNING: Airfoil " + airfoil_name[i] + " has the drag coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
            #     polar.cd[i, 0, 0] = polar.cd[i, -1, 0]
            # if abs(polar.cm[i, 0, 0] - polar.cm[i, -1, 0]) > 1.0e-5:
            #     print("WARNING: Airfoil " + airfoil_name[i] + " has the moment coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
            #     polar.cm[i, 0, 0] = polar.cm[i, -1, 0]
            # if cpmin_flag and abs(cpmin[i, 0, 0] - cpmin[i, -1, 0]) > 1.0e-5:
            #     print("WARNING: Airfoil " + airfoil_name[i] + " has the minimum pressure coefficient different between + and - pi rad. This is fixed automatically, but please check the input data.")
            #     cpmin[i, 0, 0] = cpmin[i, -1, 0]

            polar_list.append(polar)

        station_polar = []
        station_thickness = np.zeros(nStations)
        station_Ca = np.zeros((nStations, 2))
        for i in range(nStations):
            for j in range(n_af):
                if station_airfoil[i] == airfoil_name[j]:
                    station_polar.append(polar_list[j])
                    station_thickness[i] = airfoil_thickness[j]
                    station_Ca[i,:] = Ca[j,:]
                    break


        # Set discretization parameters
        nSector = getFromDict(turbine['blade'][ir], 'nSector', default=4) # number of equally spaced azimuthal positions for CCblade to compute and average over
        nr = getFromDict(turbine['blade'][ir], 'nr', dtype=int, default=20) # number of radial blade stations (or blade elements) to use
        
        grid = np.linspace(0., 1., nr, endpoint=False) + 0.5/nr # equally spaced grid along blade span, root=0 tip=1

        # grid = 0.5*(np.sin(0.5*np.pi*(np.linspace(-1, 1, nr, endpoint=False) + 0.5/nr)) + 1)

        spline = PchipInterpolator  # select spline interpolation method
        
        # spline interpolate airfoil thickness over evenly spaced element locations along span
        rthick_spline = spline(station_position, station_thickness)
        self.r_thick_interp = rthick_spline(grid)

        Ca_spline = spline(station_position, station_Ca)
        self.Ca_interp = Ca_spline(grid)
        
        # split out blade geometry info from table 
        geometry_table = np.array(turbine['blade'][ir]['geometry'])
        r_input           = geometry_table[:,0]
        self.dr = (rtip - self.Rhub)/nr
        
        # radial locations of blade elements for BEM
        # grid             =  np.sin(0.5*np.pi*(np.linspace(0, 1, nr, endpoint=False) + 0.5/nr)) 
        # self.blade_r      = np.sin(np.linspace(0, 1, nr, endpoint=False) + 0.5/nr)  
        # self.blade_r      = np.linspace(self.Rhub, rtip, nr, endpoint=False) + self.dr/2
        self.blade_r      = (rtip - self.Rhub) * grid + self.Rhub 
        
        self.blade_chord  = np.interp(self.blade_r, r_input, geometry_table[:,1])
        self.blade_theta  = np.interp(self.blade_r, r_input, geometry_table[:,2])
        blade_precurve    = np.interp(self.blade_r, r_input, geometry_table[:,3])
        blade_presweep    = np.interp(self.blade_r, r_input, geometry_table[:,4])

        if geometry_table.shape[1] == 6:
            self.pitch_axis = np.interp(self.blade_r, r_input, geometry_table[:,5])
        else:
            self.pitch_axis = np.ones_like(grid) * 0.25
        #  <<<<<< move this to beginning, then do some interpolating to unify grid and blade_r <<<<<<< and go from above 0 to below 1

        if self.r3[2] < 0:
            self.rho = turbine['rho_water']
            self.mu = turbine['mu_water']
            self.shearExp = turbine['shearExp_water']
        else:
            self.rho = turbine['rho_air']
            self.mu = turbine['mu_air']
            self.shearExp = turbine['shearExp_air']
        

        af = []
        af_flag = 0
        for i in range(nr):
            if i == nr-1:
                aoa = station_polar[-1].alpha
                Re  = station_polar[-1].Re
                cl  = station_polar[-1].cl
                cd  = station_polar[-1].cd
                cm  = station_polar[-1].cm
                af.append(CCAirfoil(aoa, [Re], cl, cd, cm, AFName=station_airfoil[-1]))
                
                break
            elif grid[i] > station_position[af_flag] and grid[i] <= station_position[af_flag+1]:
                aoa = station_polar[af_flag].alpha
                Re  = station_polar[af_flag].Re
                cl  = station_polar[af_flag].cl
                cd  = station_polar[af_flag].cd
                cm  = station_polar[af_flag].cm
                af.append(CCAirfoil(aoa, [Re], cl, cd, cm, AFName=station_airfoil[af_flag]))
            elif grid[i] > station_position[af_flag+1]:
                aoa = station_polar[af_flag+1].alpha
                Re  = station_polar[af_flag+1].Re
                cl  = station_polar[af_flag+1].cl
                cd  = station_polar[af_flag+1].cd
                cm  = station_polar[af_flag+1].cm
                af.append(CCAirfoil(aoa, [Re], cl, cd, cm, AFName=station_airfoil[af_flag+1]))

            if grid[i+1] > station_position[af_flag+1]:
                af_flag += 1

            
        # >>> There is an inconsistency between the geometric and airfoil inputs that needs to be corrected! <<<
        
        self.ccblade = CCBlade(
            self.blade_r,                        # (m) locations defining the blade along z-axis of blade coordinate system
            self.blade_chord,                    # (m) corresponding chord length at each section
            self.blade_theta,                    # (deg) corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---positive twist decreases angle of attack.
            af,                             # CCAirfoil object
            self.Rhub,                      # (m) radius of hub
            turbine['blade'][ir]['Rtip'],   # (m) radius of tip
            self.nBlades,                   # number of blades
            self.rho,                       # (kg/m^3) freestream fluid density
            self.mu,                        # (kg/m/s) dynamic viscosity of fluid
            self.precone,                   # (deg) hub precone angle
            np.degrees(self.shaft_tilt),    # (deg) hub tilt angle  
            0.0,                            # (deg) nacelle yaw angle
            self.shearExp,                  # shear exponent for a power-law wind profile across hub
            self.r3[2],                      # (m) hub height used for power-law wind profile.  U = Uref*(z/hubHt)**shearExp
            nSector,                        # number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to 1 if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
            blade_precurve,                 # (m) location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            turbine['blade'][ir]['precurveTip'],# (m) location of blade pitch axis in x-direction at the tip (analogous to Rtip)
            blade_presweep,                 # (m) location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            turbine['blade'][ir]['presweepTip'],# (m) location of blade pitch axis in y-direction at the tip (analogous to Rtip)
            tiploss=tiploss,                # if True, include Prandtl tip loss model
            hubloss=hubloss,                # if True, include Prandtl hub loss model
            wakerotation=wakerotation,      # if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
            usecd=usecd,                    # If True, use drag coefficient in computing induction factors (always used in evaluating distributed loads from the induction factors).
            derivatives=True,               # if True, derivatives along with function values will be returned for the various methods
        )
        
        # pull control gains out of dictionary
        self.setControlGains(turbine)
        
        # create a member list of blade sections, only if rotor is underwater
        if self.r3[2] + self.R_rot < 0:
            #self.bladeAirfoil2Member()
            # self.bladeGeometry2Member()
            self.bladeGeometry2BladeMember(geometry_table)
        else:
            self.bladeMemberList = []
    
    
    def setPosition(self, r6=np.zeros(6), R=None):
        '''Calculate rotor pose based on FOWT pose.
        
        Parameters
        ----------
        r6 : array, optional
            Absolute position/orientation of FOWT to which member is attached.
        '''
        
        # store rotation matrix for platform roll, pitch, yaw
        if R:
            self.R_ptfm = np.array(R)
        else:
            self.R_ptfm = rotationMatrix(*r6[3:])
        
        # Store platform heading for use with nacelle yaw
        self.platform_heading = r6[5]
        
        # Apply nacelle yaw depending on the yaw mode 
        self.setYaw()
        
        # Update RNA point locations [m] w.r.t. PRP in global orientations
        self.r_RRP_rel = np.matmul(self.R_ptfm, self.r_rel) # RNA ref point
        self.r_CG_rel = self.r_RRP_rel + self.q*self.xCG_RNA # RNA CG location
        self.r_hub_rel = self.r_RRP_rel + self.q*self.overhang # rotor hub location
        
        '''
        self.r_RRP = ? # RNA reference point
        self.r_CG  = ? # RNA CG location
        self.r_hub = ? # rotor hub coordinates in global [m]
        '''
        
        # Set absolute hub coordinate [m] for use in various aero/hydro calcs
        self.r3 = r6[:3] + self.r_hub_rel  
        
    
    def setYaw(self, yaw=None):
        ''' Sets the nacelle yaw and computes quantities related to rotor
        orientation according to the yaw mode.
        
        Parameters
        ----------
        yaw : float, optional
            Requested nacelle yaw [deg]. Relative to platform if yaw_mode=2, absolute
            heading if yaw_mode==3. Saved in self.yaw_command.
        heading : float, optional
            The heading of wind/current inflow if yaw_mode==0 [deg].
            The heading of the turbine if yaw_mode==1 [deg].
        '''
        
        # Use and save the yaw command if a new one was provided
        if not yaw == None:
            self.yaw_command = np.radians(yaw)
        
        # ----- Apply nacelle yaw offset depending on the yaw mode -----
        
        if self.yaw_mode == 0:  # use yaw value as inflow misalignment angle
            self.yaw = (self.inflow_heading - self.platform_heading 
                        + self.yaw_command)
        
        elif self.yaw_mode == 1:  # use case info
            #turb_heading = getFromDict(case, 'turbine_heading', shape=0, default=0.0)  # [rad]
            self.yaw = self.turbine_heading - self.platform_heading  # >> should we add an offset option? <<
        
        elif self.yaw_mode == 2:  # use yaw value as relative to platform
            self.yaw = self.yaw_command
            
        elif self.yaw_mode == 3: # use yaw_command value as global heading
            self.yaw = self.yaw_command - self.platform_heading
            
        else:
            raise Exception('Unsupported yaw_mode value. Must be 0, 1, or 2.')
        
        # Set turbine heading in case the user checks (redundant for yaw_mode 1)
        self.turbine_heading = self.platform_heading + self.yaw

        # Rotation matrix from platform local x to rotor axis
        R_q_rel = rotationMatrix(0, self.shaft_tilt, self.shaft_toe + self.yaw) 
        self.R_q = np.matmul(R_q_rel, self.R_ptfm)  # this one's from global x
        
        # Compute shaft axis unit vector in FOWT and global frames 
        self.q_rel = np.matmul(R_q_rel, np.array([1,0,0]) )
        self.q = np.matmul(self.R_ptfm, self.q_rel) # Write in the global frame 
        
        return self.yaw
    
    
    """
    def bladeAirfoil2Member(self, Ca_edge=0.5, Ca_flap=1.0):
        '''First iteration of a method to create RAFT members for the rotor blades (not used right now).

        Method to create members for each airfoil in the turbine blade
        To be used for added mass and buoyancy calculations of underwater turbines'''
        
        self.bladeMemberList = []
        blade_length = self.R_rot-self.Rhub
        blade_r = np.array(station_position)*blade_length

        airfoil_name_dict = [foil['name'] for foil in self.turbine['airfoils']]

        for i,af in enumerate(station_airfoil[:-1]):
            airfoil = {}        # dictionary to hold properties of blade sub-member = each airfoil
            airfoil['name'] = af+'-'+str(i+1)+'/'+str(len(station_airfoil))
            airfoil['type'] = 3

            # started a fancy method to determine the axis that the airfoil blades will have different headings about
            # ideally, I want to find a vector (r) orthogonal to the rotor axis vector (n), which has infinite solutions (r dot n = 0)
            # the way I set it up below rotates the rotor axis vector 90 degrees, but breaks down if rotor.axis[2] != 0 since the rotation metrix used here is about the z axis
            airfoil_zero_heading = np.matmul(np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]]), self.q_rel)
            airfoil['rA'] = np.array(airfoil_zero_heading)*(self.Rhub+(station_position[i]*blade_length))
            airfoil['rB'] = airfoil['rA'] + np.array(airfoil_zero_heading)*((station_position[i+1]-station_position[i])*blade_length)

            #airfoil['rA'] = rHub + np.array([0,0,self.Rhub]) + np.array([0,0,(station_position[i])*blade_length])
            #airfoil['rB'] = airfoil['rA'] + np.array([0,0, (station_position[i+1]-station_position[i])*blade_length])
            # >>>>>>>> don't need to specify direction of blade; just assume vertical and then can transform in later operations <<<<<<<<<<<<<
            airfoil['shape'] = 'rect'
            airfoil['stations'] = [0,1]

            chord = np.interp(blade_r[i], self.blade_r, self.blade_chord)
            rel_t = self.turbine["airfoils"][airfoil_name_dict.index(af)]["relative_thickness"]
            A = (np.pi/4)*chord**2 * rel_t
            sideB = A/chord     # the length of the imaginary side length of the rectange that gives the same area

            airfoil['d'] = [chord, sideB]
            #airfoil['d'] = np.interp(blade_r[i:i+2], self.blade_r, self.blade_chord)
            airfoil['gamma'] = np.interp(blade_r[i], self.blade_r, self.blade_theta)
            airfoil['potMod'] = False

            airfoil['Cd'] = 0.0
            if 'added_mass_coeff' in self.turbine["airfoils"][airfoil_name_dict.index(af)]:
                added_mass_coeff = self.turbine["airfoils"][airfoil_name_dict.index(af)]['added_mass_coeff']
            else:
                added_mass_coeff = [Ca_edge, Ca_flap]
            airfoil['Ca'] = added_mass_coeff 
            #airfoil['Ca'] = self.turbine["airfoils"][airfoil_name_dict.index(af)]["added_mass_coeff"]
            airfoil['CdEnd'] = 0.0
            airfoil['CaEnd'] = 0.0
        
            airfoil['t'] = 0.01
            airfoil['rho_shell'] = 1850

            self.bladeMemberList.append(Member(airfoil, len(self.w)))
    """

    # def bladeAirfoil2Member(self, Ca_edge=0.5, Ca_flap=1.0):
    #     '''First iteration of a method to create RAFT members for the rotor blades (not used right now).

    #     Method to create members for each airfoil in the turbine blade
    #     To be used for added mass and buoyancy calculations of underwater turbines'''
        
    #     self.bladeMemberList = []
    #     blade_length = self.R_rot-self.Rhub
    #     # blade_r = np.array(self.)*blade_length

    #     airfoil_name_dict = [foil['name'] for foil in self.turbine['airfoils']]

    #     for i,af in enumerate(self.balde_r[:-1]):
    #         airfoil = {}        # dictionary to hold properties of blade sub-member = each airfoil
    #         airfoil['name'] = af+'-'+str(i+1)+'/'+str(len(station_airfoil))
    #         airfoil['type'] = 3

    #         # started a fancy method to determine the axis that the airfoil blades will have different headings about
    #         # ideally, I want to find a vector (r) orthogonal to the rotor axis vector (n), which has infinite solutions (r dot n = 0)
    #         # the way I set it up below rotates the rotor axis vector 90 degrees, but breaks down if rotor.axis[2] != 0 since the rotation metrix used here is about the z axis
    #         airfoil_zero_heading = np.matmul(np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]]), self.q_rel)
    #         airfoil['rA'] = np.array(airfoil_zero_heading)*(self.Rhub+(station_position[i]*blade_length))
    #         airfoil['rB'] = airfoil['rA'] + np.array(airfoil_zero_heading)*((station_position[i+1]-station_position[i])*blade_length)

    #         #airfoil['rA'] = rHub + np.array([0,0,self.Rhub]) + np.array([0,0,(station_position[i])*blade_length])
    #         #airfoil['rB'] = airfoil['rA'] + np.array([0,0, (station_position[i+1]-station_position[i])*blade_length])
    #         # >>>>>>>> don't need to specify direction of blade; just assume vertical and then can transform in later operations <<<<<<<<<<<<<
    #         airfoil['shape'] = 'rect'
    #         airfoil['stations'] = [0,1]

    #         chord = np.interp(blade_r[i], self.blade_r, self.blade_chord)
    #         rel_t = self.turbine["airfoils"][airfoil_name_dict.index(af)]["relative_thickness"]
    #         A = (np.pi/4)*chord**2 * rel_t
    #         sideB = A/chord     # the length of the imaginary side length of the rectange that gives the same area

    #         airfoil['d'] = [chord, sideB]
    #         #airfoil['d'] = np.interp(blade_r[i:i+2], self.blade_r, self.blade_chord)
    #         airfoil['gamma'] = np.interp(blade_r[i], self.blade_r, self.blade_theta)
    #         airfoil['potMod'] = False

    #         airfoil['Cd'] = 0.0
    #         if 'added_mass_coeff' in self.turbine["airfoils"][airfoil_name_dict.index(af)]:
    #             added_mass_coeff = self.turbine["airfoils"][airfoil_name_dict.index(af)]['added_mass_coeff']
    #         else:
    #             added_mass_coeff = [Ca_edge, Ca_flap]
    #         airfoil['Ca'] = added_mass_coeff 
    #         #airfoil['Ca'] = self.turbine["airfoils"][airfoil_name_dict.index(af)]["added_mass_coeff"]
    #         airfoil['CdEnd'] = 0.0
    #         airfoil['CaEnd'] = 0.0
        
    #         airfoil['t'] = 0.01
    #         airfoil['rho_shell'] = 1850

    #         self.bladeMemberList.append(Member(airfoil, len(self.w)))
    
    def bladeGeometry2BladeMember(self, geometry_table=np.ndarray, pitch=0):
        '''A function to create RAFT BladeMembers based on rotor blades (Yang currently used). 
        slightly different from the original version bladeGeometry2Member

        Method to create members for each "node" that is specified in turbine['blade']['geometry']
        To be used for added mass and buoyancy calculations of underwater turbines'''

        self.bladeMemberList:Iterable[BladeMember] = []
        from raft.helpers import inerpStr

        if geometry_table.shape[1] == 6:
            pitch_axis = geometry_table[:,5]
        else:
            pitch_axis = np.ones_like(geometry_table[:, 0]) * 0.25
        
        r = np.zeros((geometry_table.shape[0], 3))
        r_section = []
        af_name = [af.AFName for af in self.ccblade.af]
        af_list = inerpStr(geometry_table[:, 0], self.ccblade.r, af_name)

        for i in range(len(geometry_table[:, 0])):
                  
            file = os.path.join(raft_dir, f"designs/airfoil/coordinate/{af_list[i]}.txt")
            coord = readCoordinate(file)

            if coord[0, 1] == coord[-1, 1]:
                print(f"Warnig: y coordinate of coord[0, 1] and coord[-1, 1] should not be the same, both will be deleted")
                coord = coord[1:-1]
            
            rpts = []

            n_p = coord.shape[0]  # number of points in airfoil coordinate file
            r_b = np.array([geometry_table[i, 3], geometry_table[i, 4], geometry_table[i, 0]])     # airfoil center in coordinates in blade-aligned coordinate system

            R = rotationMatrix(np.deg2rad(self.azimuths[0]), -np.rad2deg(self.precone), -geometry_table[i, 2]-pitch)    # R_z(-twist-pitch) -> R_y(-cone) -> R_x(azimuth) 
            for j in range(n_p):

                x = coord[j, 1] * geometry_table[i, 1] + r_b[0]    # x coordinate in blade-aligned coordinate system
                y = (coord[j, 0]-pitch_axis[i]) * geometry_table[i, 1] + r_b[1]    # y coordinate in blade-aligned coordinate system
                z = r_b[2]                                          # z coordinate in blade-aligned coordinate system

                rpts_rel = np.matmul(self.R_q, np.matmul(R, np.array([x,y,z])))
                rpts.append([rpts_rel[0]+self.r3[0], rpts_rel[1]+self.r3[1], rpts_rel[2]+self.r3[2]])
            
                                                                                    # coordinates in azimuth-aligned coordinate system
            rc = np.matmul(self.R_q, np.matmul(R, r_b))                         # coordinates in absolute coordinate system w.r.t. hub
            r[i]  = np.array([rc[0]+self.r3[0], rc[1]+self.r3[1], rc[2]+self.r3[2]]) # absolute coordinate
            r_section.append(rpts)
            
        for i in range(len(geometry_table[:, 0])-1):    
            
            blademem = {}
            blademem['name'] = i
            blademem['type'] = 3
            
            blademem['rA'] = r[i]
            blademem['rB'] = r[i+1]

            blademem['rSectionA'] = r_section[i]
            blademem['rSectionB'] = r_section[i+1]

            blademem['shape'] = 'rect'
            blademem['stations'] = [0,1]

            chord = geometry_table[i, 1]
            
            r_thick_interp = np.interp(geometry_table[:, 0], self.blade_r, self.r_thick_interp)
            rel_thick = r_thick_interp[i]
            
            area = (np.pi/4)*chord**2 * rel_thick
            rect_thick = area/chord  # thickness of rectange with same chord length to achieve same cross sectional area
            blademem['d'] = [[chord, rect_thick],[chord, rect_thick]]

            blademem['gamma'] = geometry_table[i, 2] + pitch

            blademem['potMod'] = False

            blademem['Cd'] = 0.0
            Ca_interp = np.stack((np.interp(geometry_table[:, 0], self.blade_r, self.Ca_interp[:, 0]), 
                                  np.interp(geometry_table[:, 0], self.blade_r, self.Ca_interp[:, 1])), axis=-1)
            blademem['Ca'] = Ca_interp[i]    
            blademem['CdEnd'] = 0.0
            blademem['CaEnd'] = 0.0
        
            blademem['t'] = 0.01
            blademem['rho_shell'] = 1850

            from raft.raft_member_yang import BladeMember

            self.bladeMemberList.append(BladeMember(blademem, len(self.w)))
        
        # self.nodes = np.zeros([int(self.nBlades), len(self.bladeMemberList)+1, 3])      # array to hold xyz positions of each node along a blade for each blade (filled in later)

    def bladeGeometry2Member(self):
        '''Second iteration of a function to create RAFT members based on rotor blades (is currently used).

        Method to create members for each "node" that is specified in turbine['blade']['geometry']
        To be used for added mass and buoyancy calculations of underwater turbines'''

        self.bladeMemberList:Iterable[BladeMember] = []

        for i in range(len(self.blade_r)-1):
            blademem = {}
            blademem['name'] = i
            blademem['type'] = 3

            airfoil_zero_heading = np.matmul(np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]]), self.q_rel) # see comments in bladeAirfoil2Member()
            blademem['rA'] = np.array(airfoil_zero_heading) * (self.blade_r[i] - self.dr/2)
            blademem['rB'] = np.array(airfoil_zero_heading) * (self.blade_r[i] + self.dr/2)

            blademem['shape'] = 'rect'
            blademem['stations'] = [0,1]

            chord = self.blade_chord[i]
            rel_thick = self.r_thick_interp[i]
            area = (np.pi/4)*chord**2 * rel_thick
            rect_thick = area/chord  # thickness of rectange with same chord length to achieve same cross sectional area
            blademem['d'] = [[chord, rect_thick],[chord, rect_thick]]

            blademem['gamma'] = self.blade_theta[i]

            blademem['potMod'] = False

            blademem['Cd'] = 0.0
            blademem['Ca'] = self.Ca_interp[i,:]            
            blademem['CdEnd'] = 0.0
            blademem['CaEnd'] = 0.0
        
            blademem['t'] = 0.01
            blademem['rho_shell'] = 1850

            self.bladeMemberList.append(Member(blademem, len(self.w)))
        
        self.nodes = np.zeros([int(self.nBlades), len(self.bladeMemberList)+1, 3])      # array to hold xyz positions of each node along a blade for each blade (filled in later)


    def getBladeMemberPositions(self, azimuth, r_OG):
        ''' Returns the node positions of blade members as it is rotated by an azimuth angle about the rotor's axis.
        rOG is a matrix of n number of rows and 3 columns, where each row is a position vector that needs rotating'''

        # create rotation matrix based on the rotor's axis (default axis=[1,0,0])
        c = np.cos(np.deg2rad(azimuth))
        s = np.sin(np.deg2rad(azimuth))
        a = self.q_rel  # each rotor is given a default axis of rotation about the x-direction
        R = np.array([[c + a[0]**2*(1-c), a[0]*a[1]*(1-c)-a[2]*s, a[0]*a[2]*(1-c)+a[1]*s],
                        [a[1]*a[0]*(1-c)+a[2]*s, c + a[1]**2*(1-c), a[1]*a[2]*(1-c)-a[0]*s],
                        [a[2]*a[0]*(1-c)-a[1]*s, a[2]*a[1]*(1-c)+a[0]*s, c + a[2]**2*(1-c)]])

        # find the new node positions of the blade member
        r_new = np.zeros_like(r_OG)
        for i in range(r_OG.shape[0]):
            r_from_Zhub = np.matmul(R, r_OG[i,:])   # wrt to the hub center
            r_new[i,:] = r_from_Zhub + self.r_hub    # wrt to the global coordinates

        return r_new


    def calcHydroConstants(self, dgamma=0, rho=1025, g=9.81):
        '''Compute hydrodynamic added mass and inertial excitation coefficients
        for the rotor as a whole. Stores and returns results in local reference
        frame about hub location.
        
        Parameters
        ----------
        dgamma : float
            Blade pitch angle to be applied to the rotor.
        
        Returns
        -------
        A_hydro, I_hydro : 3x3 matrices
            Hydrodynamic added mass and inertial excitation matrices.
        '''
        
        A_hydro = np.zeros([6,6])
        I_hydro = np.zeros([6,6])
        
        # --- whole-rotor members approach ---
        for i,mem in enumerate(self.bladeMemberList):

            rOG = np.array([mem.rA0, mem.rB0])      # save the original position of the blade members

            for theta in self.azimuths:             # for each blade member, repeat the process for each heading

                # find the end nodes of the blade member about the global coordinates (e.g,, if rA_OG = [0,0,0] and rB_OG=[0,1,0], and heading=90, then rA=[0,0,0] and rB=[0,0,1])
                rUpdated = self.getBladeMemberPositions(theta, rOG) # blade node coords relative to hub
                mem.rA0 = rUpdated[0]
                mem.rB0 = rUpdated[-1]
                
                # apply a blade pitch angle to every blade member if applicable
                mem.gamma = mem.gamma + dgamma

                # run calcOrientation to ensure the p1 and p2 axes of the members are oriented the way they should be
                mem.setPosition()
                
                # compute hydro added mass and inertial excitation terms relative to hub
                A_hydro_i, I_hydro_i = mem.calcHydroConstants(sum_inertia=True, rho=rho, g=g)
                
                A_hydro += A_hydro_i 
                I_hydro += I_hydro_i
            
            mem.rA0 = rOG[0]    # reset end positions of each blade member after rotating
            mem.rB0 = rOG[1]
        
        # --- save hydro matrices (these are in global orientations) ---
        self.A_hydro = A_hydro
        self.I_hydro = I_hydro
        
        return A_hydro, I_hydro


    def calcCavitation(self, case, azimuth=0, clearance_margin=1.0, Patm=101325, Pvap=2500, error_on_cavitation=False):
        ''' Method to calculate the cavitation number of the rotor
        (wind speed (m/s), rotor speed (RPM), pitch angle (deg), azimuth (deg))

        Can later move Patm and Pvap to some kind of input file
        '''
        # -------------- calculate worst case clearance below waterline (less precise) ---------------
        # calculate the worst-case scenario depth below the free surface where cavitation can occur
        if self.r3[2] < 0:
            clearance = self.r3[2] + self.R_rot
        else:
            raise ValueError("Hub Depth must be below the water surface to calculate cavitation")
        # add a margin to the depth clearance (either by user input or based on platform motions)
        clearance = clearance*clearance_margin
        
        # >>> note: above currently not used <<<
        
        #--------------- calculate clearance (depth) of each node of each blade (more precise) --------------
        # collect minimum pressure coefficient values
        cpmin = self.cpmin_interp       # array of size [len(bladeMemberList), len(self.aoa), 1] where each row is the cpmin as a function of aoa (columns)
        
        # set wind speed, rotor speed, and blade pitch angle based on wind speed an turbine inputs
        Uhub = case['current_speed']
        Omega_rpm = np.interp(Uhub, self.Uhub, self.Omega_rpm)  # rotor speed [rpm]
        pitch_deg = np.interp(Uhub, self.Uhub, self.pitch_deg)  # blade pitch angle [deg]

        # create array to store cavitation values for each node for each blade
        cav_check = np.zeros([len(self.azimuths), len(self.blade_r)])

        # calculate the critial sigma caviatation parameter for each blade node and compare to the sigma_l caviatation parameter to determine if cavitation occurs
        for a,azi in enumerate(self.azimuths):      # do this for each blade (aoa and relative velocity change for different blade azimuth angles)

            loads, derivs = self.ccblade.distributedAeroLoads(Uhub, Omega_rpm, pitch_deg, azi)  # run CCBlade with variable azimuth angles
            vrel = loads["W"]       # pull out the relative velocity at each node along the blade at that azimuth angle
            aoa = loads["alpha"]    # pull out the angle of attack at each node along the blade at that azimuth angle
            
            for n in range(len(vrel)):      # for each blade node

                # find the minimum pressure coefficient at that node at the given angle of attack
                cpmin_node = np.interp(aoa[n], self.aoa, cpmin[n,:,0])

                # extract the depth of the node using the node position array
                clearance = self.nodes[a, n, 2]     # a=which blade, n=which node, 2=z-position=depth
                
                # calculate the critial sigma cavitation parameter
                sigma_crit = (Patm + self.ccblade.rho*9.81*abs(clearance) - Pvap)/(0.5*self.ccblade.rho*vrel[n]**2)

                # if sigma_crit is less than sigma_l (sigma_l = -cpmin), then cavitation occurs
                if error_on_cavitation:
                    if sigma_crit < -cpmin_node:
                        raise ValueError(f"Cavitation occured at node {n} (first node = 0)")
                
                cav_check[a,n] = sigma_crit + cpmin_node         # if this value is negative, then cavitation occurs (sigma_crit - sigma_l < 0 -> cav occurs; sigma_l = -cpmin_node)

        if np.any(cav_check < 0.0):
            print("WARNING: Cavitation check was run and found a blade node that has cavitation occuring")

        return cav_check


    def runCCBlade(self, U0, tilt=0, yaw_misalign=0):
        '''This performs a single CCBlade evaluation at specified conditions.
        
        U0
            Freestream flow speed [m/s].
        tilt
            tilt angle due to both shaft tilt and platform tilt [rad]
        yaw_misalign
            turbine yaw misalignment angle [rad]
            
        In future could add options to specify rotor speed and blade pitch values
        that override the default scheduled values, for controls applications.
        '''
        
        # apply inflow speed gain factor (for any confinement/blockage effects)
        Uhub = U0*self.speed_gain
        
        # find turbine operating point at the provided wind speed
        Omega_rpm = np.interp(Uhub, self.Uhub, self.Omega_rpm)  # rotor speed [rpm]
        pitch_deg = np.interp(Uhub, self.Uhub, self.pitch_deg)  # blade pitch angle [deg]
        
        # Adjust rotor angles based on provided info
        # Note: this adjusts internal CCBlade angles (which are in radians)
        self.ccblade.tilt = tilt
        self.ccblade.yaw  = yaw_misalign
        
        # evaluate aero loads and derivatives with CCBlade
        loads, derivs = self.ccblade.evaluate(Uhub, Omega_rpm, pitch_deg, coefficients=True)

        # organize and save the relevant outputs...
        self.U_case         = Uhub
        self.Omega_case     = Omega_rpm
        self.aero_torque    = loads["Q"][0]
        self.aero_power     = loads["P"][0]
        self.aero_thrust    = loads["T"][0]
        self.pitch_case     = pitch_deg

        outputs = {}
        
        outputs["P"] = loads["P"]
        outputs["Mb"] = loads["Mb"]
        outputs["CP"] = loads["CP"]
        outputs["CMb"] = loads["CMb"]
        outputs["Fhub"] = np.array( [loads["T" ][0], loads["Y"  ][0], loads["Z"  ][0]])
        outputs["Mhub"] = np.array( [loads["Q" ][0], loads["My" ][0], loads["Mz" ][0]])
        outputs["CFhub"] = np.array([loads["CT"][0], loads["CY" ][0], loads["CZ" ][0]])
        outputs["CMhub"] = np.array([loads["CQ"][0], loads["CMy"][0], loads["CMz"][0]])

        # We might want to supress this print statement? Or add a verbosity option?
        print(f"Wind speed: {Uhub:.2f} m/s, Omega: {Omega_rpm:.2f} rpm, Cp: {loads['CP'][0]:4.3f}, T: {loads['T'][0]/1e3:.0f} kN")
        
        # save select derivatives
        J={} # Jacobian/derivatives

        dP = derivs["dP"]
        J["P", "r"] = dP["dr"]

        dQ = derivs["dQ"]
        J["Q","Uhub"]      = np.atleast_1d(np.diag(dQ["dUinf"]))
        J["Q","pitch_deg"] = np.atleast_1d(np.diag(dQ["dpitch"]))
        J["Q","Omega_rpm"] = np.atleast_1d(np.diag(dQ["dOmega"]))

        dT = derivs["dT"]
        J["T","Uhub"]      = np.atleast_1d(np.diag(dT["dUinf"]))
        J["T","pitch_deg"] = np.atleast_1d(np.diag(dT["dpitch"]))
        J["T","Omega_rpm"] = np.atleast_1d(np.diag(dT["dOmega"]))
        
        self.J = J
        return loads, derivs


    def setControlGains(self,turbine):
        '''
        Use flipped sign version of ROSCO
        '''

        # Convert gain-scheduling wrt pitch to wind speed, Add zero gains for parked "control"
        pc_angles = np.array(turbine['pitch_control']['GS_Angles']) * rad2deg
        self.kp_0 = np.interp(self.pitch_deg,pc_angles,turbine['pitch_control']['GS_Kp'],left=0,right=0)
        self.ki_0 = np.interp(self.pitch_deg,pc_angles,turbine['pitch_control']['GS_Ki'],left=0,right=0)
        self.k_float = -turbine['pitch_control']['Fl_Kp']

        # Torque control
        self.kp_tau = -turbine['torque_control']['VS_KP']
        self.ki_tau = -turbine['torque_control']['VS_KI']
        self.Ng     = turbine['gear_ratio']
            


    def calcAero(self, case, current=False, display=0):
        '''Calculates stiffness, damping, added mass, and excitation coefficients
        from rotor aerodynamics coupled with turbine controls.
        Results are w.r.t. the hub coordinate on the nacelle reference frame (may be yawed), 
        but written in the global reference frame.
        Currently returning 6 DOF mean loads, but other terms are just hub fore-aft scalars.
        '''
        
        # added mass, damping, excitation, mean force arrays to be filled in (6 DOF)
        self.a = np.zeros([6,6,self.nw])
        self.b = np.zeros([6,6,self.nw])
        self.f = np.zeros([6  ,self.nw], dtype=complex)
        self.f0 = np.zeros(6)
    
        # get inflow speed and heading [deg]
        if current:
            speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
            heading = getFromDict(case, 'current_heading', shape=0, default=0.0)
        else:
            speed = getFromDict(case, 'wind_speed', shape=0, default=10)
            heading = getFromDict(case, 'wind_heading', shape=0, default=0.0)
        
        # Store case info for potential use when setting the turbine yaw
        self.inflow_heading = np.radians(heading)
        self.turbine_heading = np.radians(getFromDict(case, 'turbine_heading', 
                                          shape=0, default=0.0))
        
        self.setYaw()
                
        # rotor inflow misalignment heading and tilt for CCBlade [rad]
        yaw_misalign = np.arctan2(self.q[1], self.q[0]) - self.inflow_heading
        turbine_tilt = np.arctan2(self.q[2], np.hypot(self.q[0], self.q[1]))
        # >>> should double check signs <<<
        
        # call CCBlade
        loads, derivs = self.runCCBlade(speed, tilt=turbine_tilt,
                                        yaw_misalign=yaw_misalign)

        # ----- Process derivatives of interest -----
        dT_dU  = np.atleast_1d(np.diag(derivs["dT"]["dUinf"]))
        dT_dOm = np.atleast_1d(np.diag(derivs["dT"]["dOmega"])) / rpm2radps
        dT_dPi = np.atleast_1d(np.diag(derivs["dT"]["dpitch"])) * rad2deg
        dQ_dU  = np.atleast_1d(np.diag(derivs["dQ"]["dUinf"]))
        dQ_dOm = np.atleast_1d(np.diag(derivs["dQ"]["dOmega"])) / rpm2radps
        dQ_dPi = np.atleast_1d(np.diag(derivs["dQ"]["dpitch"])) * rad2deg
        # note: orientation corrections still need to be applied on these! <<<


        # ----- Process steady rotor forces and moments -----
        
        # Set up vectors in axis frame. Assuming CCBlade forces (but not 
        # moments) are relative to the rotor axis
        forces_axis = np.array([loads["T"][0], loads["Y"][0], loads["Z" ][0]])
        moments_axis = np.array([loads["My"][0], loads["Q"][0], loads["Mz"][0]])        
        
        # Rotate forces and moments to be relative to global orientation (but still wrt hub)
        self.f0[:3] = np.matmul(self.R_q, forces_axis)
        self.f0[3:] = np.matmul(self.R_q, moments_axis)
        
        
        # ----- Dynamic rotor forces and reaction matrices -----
        # calculate rotor-averaged turbulent wind spectrum
        _,_,_,S_rot = self.IECKaimal(case, current=current)   # PSD [(m/s)^2/rad]
        
        # convert from power spectral density to complex amplitudes (FFT)
        self.V_w = np.array(np.sqrt(S_rot), dtype=complex)  # Is there a factor of 2 missing here?

        # Do we need to worry about scaling by dot prod of rotor axis and
        # inflow direction?  *np.cos(turbine_tilt)*np.cos(yaw_misalign) <<<

        # Rotate rotor hydro inertial excitation matrix into global frame
        if current:
            I_hydro = rotateMatrix6(self.I_hydro, self.R_q)

        # no-control option
        if self.aeroServoMod == 1:  

            # Added mass matrix is empty
            a_inflow = np.zeros([6,6,len(self.w)])
            
            # Damping matrix has just windspeed-thrust derivative
            b_inflow = np.zeros([6,6,len(self.w)]) 
            b_inflow[0,0,:] = dT_dU
            
            # Excitation vector
            f_inflow = np.zeros([6,len(self.w)], dtype=np.complex_)
            f_inflow[0,:] = dT_dU*self.V_w
            
            # Rotate to global orientations
            self.a = rotateMatrix6(a_inflow, self.R_q)
            self.b = rotateMatrix6(b_inflow, self.R_q)
            self.f[:3,:] = np.matmul(self.R_q, f_inflow[:3,:])
            #self.f[3:,:] = np.matmul(self.R_q, f_inflow[3:,:]) <<< moment excitation is zero for now
            
            
        # control option
        elif self.aeroServoMod == 2:  
        
            # include added mass somewhere?  and maybe even effect of inertial excitation on control?
        
            # Pitch control gains at the inflow speed (flip sign due to ROSCO convention)
            self.kp_beta    = -np.interp(speed, self.Uhub, self.kp_0) 
            self.ki_beta    = -np.interp(speed, self.Uhub, self.ki_0) 

            # Torque control gains, need to get these from somewhere
            kp_tau = self.kp_tau * (self.kp_beta == 0)  #     -38609162.66552     ! VS_KP				- Proportional gain for generator PI torque controller [1/(rad/s) Nm]. (Only used in the transitional 2.5 region if VS_ControlMode =/ 2)
            ki_tau = self.ki_tau  * (self.ki_beta == 0)   #    -4588245.18720      ! VS_KI	
            
            a_aer = np.zeros_like(self.w)
            b_aer = np.zeros_like(self.w)
            C   = np.zeros_like(self.w,dtype=np.complex_)
            C2  = np.zeros_like(self.w,dtype=np.complex_)
            D   = np.zeros_like(self.w,dtype=np.complex_)
            E   = np.zeros_like(self.w,dtype=np.complex_)

            # Roots of characteristic equation, helps w/ debugging
            # p = np.array([-self.I_drivetrain, (dQ_dOm + self.kp_beta * dQ_dPi - self.Ng * kp_tau), self.ki_beta* dQ_dPi - self.Ng * ki_tau])
            # r = np.roots(p)

            for iw, omega in enumerate(self.w):
                
                # Denominator of control transfer function
                D[iw] = self.I_drivetrain * omega**2 + (dQ_dOm + self.kp_beta * dQ_dPi - self.Ng * kp_tau) * 1j * omega + self.ki_beta* dQ_dPi - self.Ng * ki_tau

                # control transfer function
                C[iw] = 1j * omega * (dQ_dU - self.k_float * dQ_dPi / self.r3[2]) / D[iw]

                # Thrust transfer function
                E[iw] = ((dT_dOm + self.kp_beta * dT_dPi) * 1j * omega + self.ki_beta * dT_dPi )

                # alternative for debugging
                C2[iw] = C[iw] / (1j * omega)

                # Complex aero damping
                T = 1j * omega * (dT_dU - self.k_float * dT_dPi / self.r3[2]) - ( E[iw] * C[iw])
                
                # Aerodynamic coefficients
                a_aer[iw] = -(1/omega**2) * np.real(T)
                b_aer[iw] = (1/omega) * np.imag(T)
            
            # Save transfer functions required for output
            self.C = C
            
            # calculate wind excitation force/moment spectra
            T_0 = loads["T" ][0]
            T_w1 = dT_dU * self.V_w
            T_w2 = (E * C * self.V_w) / (1j * self.w) * (-1)  # mhall: think this needs the sign reversal

            T_ext = T_w1 + T_w2


            # --- new approach ---
            
            
            # transfer function from torque to thrust                
            H_QT = ((dT_dOm + self.kp_beta*dT_dPi)*1j*self.w + self.ki_beta*dT_dPi) / (
                   self.I_drivetrain*self.w**2 + (dQ_dOm + self.kp_beta*dQ_dPi - self.Ng*kp_tau)*1j*self.w + self.ki_beta*dQ_dPi - self.Ng*ki_tau )

            # save excitation coefficient
            self.c_exc = dT_dU - H_QT*dQ_dU

            f2 = (dT_dU - H_QT*dQ_dU) * self.V_w  # excitation force
            b2 = np.real(  dT_dU - self.k_float*dT_dPi - H_QT*(dQ_dU - self.k_float*dQ_dPi)             )  # damping
            a2 = np.real( (dT_dU - self.k_float*dT_dPi - H_QT*(dQ_dU - self.k_float*dQ_dPi))/(1j*self.w))  # added mass

            # without nacelle feedback
            b3 = np.real(  dT_dU - H_QT*dQ_dU             )  # damping
            a3 = np.real( (dT_dU - H_QT*dQ_dU)/(1j*self.w))  # added mass

            # Add hydrodynamic inertial excitation for underwater rotors
            # (thrust only, neglecting rotor dynamics)
            #if current:
            #    f2 += self.I_hydro[0,0] * 1j*self.w*self.V_w

            if display > 1:               
                fig,ax = plt.subplots(4,1,sharex=True)
                ax[0].plot(self.w/2.0/np.pi, self.V_w);  ax[0].set_ylabel('U (m/s)') 
                ax[1].plot(self.w/2.0/np.pi, T_w1    );  ax[1].set_ylabel('T_w1') 
                ax[2].plot(self.w/2.0/np.pi, np.real(T_w2),'k')
                ax[2].plot(self.w/2.0/np.pi, np.imag(T_w2),'k:'); ax[2].set_ylabel('T_w2') 
                ax[3].plot(self.w/2.0/np.pi, np.real(T_w1+T_w2),'k')
                ax[3].plot(self.w/2.0/np.pi, np.imag(T_w1+T_w2),'k:'); ax[3].set_ylabel('T_w2+T_w2') 
                ax[3].set_xlabel('f (Hz)') 
                
                
                fig,ax = plt.subplots(4,1,sharex=True)
                ax[0].plot(self.w/2.0/np.pi, self.V_w);  ax[0].set_ylabel('U (m/s)') 
                ax[1].plot(self.w/2.0/np.pi, T_w1 , 'g--')
                ax[1].plot(self.w/2.0/np.pi, T_w2 , 'g:')
                ax[1].plot(self.w/2.0/np.pi, T_ext, 'g')
                ax[1].plot(self.w/2.0/np.pi, dT_dU*self.V_w, 'k--')
                ax[1].plot(self.w/2.0/np.pi,-H_QT*dQ_dU*self.V_w , 'k:')
                ax[1].plot(self.w/2.0/np.pi, f2        , 'k'  );  ax[1].set_ylabel('F') 
                ax[2].plot(self.w/2.0/np.pi, b_aer     , 'g')
                ax[2].plot(self.w/2.0/np.pi, b3        , 'b')
                ax[2].plot(self.w/2.0/np.pi, b2        , 'k--');  ax[2].set_ylabel('B') 
                ax[3].plot(self.w/2.0/np.pi, a_aer     , 'g')
                ax[3].plot(self.w/2.0/np.pi, a3        , 'b')
                ax[3].plot(self.w/2.0/np.pi, a2        , 'k--');  ax[3].set_ylabel('A') 
                ax[3].set_xlabel('f (Hz)') 

                plt.show()
                
            # Rotate to global orientations
            for iw in range(self.nw):
                self.a[:3,:3, iw] = rotateMatrix3(np.diag([a2[iw],0,0]), self.R_q)
                self.b[:3,:3, iw] = rotateMatrix3(np.diag([b2[iw],0,0]), self.R_q)
                self.f[:3,    iw] = np.matmul(self.R_q, np.array([f2[iw],0,0]))
                # Above is only forces for now. Moments can be added in future.
        
        """ # Caltured in FOWT.calcHydroExcitation()
        # Add hydrodynamic inertial excitation for underwater rotors
        if current:
            self.f[0,:] += self.I_hydro[0,0] * 1j*self.w*self.V_w  # <<< this should have a rotation applied
            breakpoint()
        """        
        
        return self.f0, self.f, self.a, self.b #  B_aero, C_aero, F_aero0, F_aero
        
        
    def plot(self, ax, r_ptfm=np.array([0,0,0]), azimuth=0, color='k', 
             airfoils=False, draw_circle=False,
             plot2d=False, Xuvec=[1,0,0], Yuvec=[0,0,1], zorder=2):
        '''Draws the rotor on the passed axes, considering optional platform 
        offset and rotation matrix, and rotor azimuth angle.
        
        Parameters
        ----------        
        plot2d: bool
            If true, produces a 2d plot on the axes defined by Xuvec and Yuvec. 
            Otherwise produces a 3d plot (default).
        draw_circle : bool
            If true, draw circle of rotor circumference.
        '''

        Xuvec = np.array(Xuvec)
        Yuvec = np.array(Yuvec)
        
        # support self color option
        if color == 'self':
            color = self.color  # attempt to allow custom colors
        

        # ----- blade geometry ----------

        m = len(self.ccblade.chord)

        # lists to be filled with coordinates for plotting
        X = []
        Y = []
        Z = []        
        
        # generic airfoil for now
        afx = np.array([ 0.0 , -0.16, 0.0 ,  0.0 ])
        afy = np.array([-0.25,  0.  , 0.75, -0.25])
        npts = len(afx)
        
        # should add real airfoil shapes, and twist     
        for i in range(m):
            for j in range(npts):
                X.append(self.ccblade.chord[i]*afx[j])
                Y.append(self.ccblade.chord[i]*afy[j])
                Z.append(self.ccblade.r[i])            
                
        P = np.array([X, Y, Z])  # coordinates of blade chord sections
        
        # ----- rotation matrices ----- 
        # (blade pitch would be a -rotation about local z)
        R_precone = rotationMatrix(0, -self.ccblade.precone, 0)  
        R_azimuth = [rotationMatrix(azimuth + azi, 0, 0) for azi in (2*np.pi/self.nBlades)*np.arange(self.nBlades)]
        R_tilt    = rotationMatrix(0, self.shaft_tilt, 0)   # # define x as along shaft downwind, y is same as ptfm y
        
        # ----- transform coordinates -----
        for ib in range(self.nBlades):
        
            P2 = np.matmul(R_precone, P)
            P2 = np.matmul(R_azimuth[ib], P2)  # rotate around shaft
            P2 = np.matmul(self.R_q, P2)  # rotate to actual rotor orientation
            P2 = P2 + self.r3[:,None]  # translate from PRP to absolute hub location
            
            
            if plot2d:  # new 2d plotting option
                
                # apply any 3D to 2D transformation here to provide desired viewing angle
                Xs2d = np.matmul(Xuvec, P2)
                Ys2d = np.matmul(Yuvec, P2)
                
                if airfoils:
                    for ii in range(m-1):
                        ax.plot(Xs2d[npts*ii:npts*(ii+1)], Ys2d[npts*ii:npts*(ii+1)], color=color, lw=0.4)  
                # draw outline
                ax.plot(Xs2d[0:-1:npts], Ys2d[0:-1:npts], color=color, lw=0.4, zorder=zorder) # leading edge  
                ax.plot(Xs2d[2:-1:npts], Ys2d[2:-1:npts], color=color, lw=0.4, zorder=zorder)  # trailing edge

            
            else:  # normal 3d case
                # drawing airfoils                            
                if airfoils:
                    for ii in range(m-1):
                        ax.plot(P2[0, npts*ii:npts*(ii+1)], P2[1, npts*ii:npts*(ii+1)], P2[2, npts*ii:npts*(ii+1)], color=color, lw=0.4)  
                # draw outline
                ax.plot(P2[0, 0:-1:npts], P2[1, 0:-1:npts], P2[2, 0:-1:npts], color=color, lw=0.4, zorder=zorder) # leading edge  
                ax.plot(P2[0, 2:-1:npts], P2[1, 2:-1:npts], P2[2, 2:-1:npts], color=color, lw=0.4, zorder=zorder)  # trailing edge

        # ----- Also draw a circle -----
        if draw_circle:
            r = self.ccblade.r[-1]
            
            # lists to be filled with coordinates for plotting
            X=[]
            Y=[]
            Z=[]
            
            n = 24                      # number of sides for a circle
            for i in range(n+1):
                y = np.cos(float(i)/float(n)*2.0*np.pi)    # x coordinates of a unit circle
                z = np.sin(float(i)/float(n)*2.0*np.pi)    # y

                X.append(0)
                Y.append(r*y)
                Z.append(r*z)

            Pcirc = np.vstack([X, Y, Z])  # rotor circle coordinates
            P2 = np.matmul(self.R_q, Pcirc)
            P2 = P2 + self.r3[:,None]  # translate from PRP to absolute hub location 
            
            if plot2d:  # new 2d plotting option
                # apply any 3D to 2D transformation here to provide desired viewing angle
                Xs2d = np.matmul(Xuvec, P2)
                Ys2d = np.matmul(Yuvec, P2)
                ax.plot(Xs2d, Ys2d, color=color, lw=0.4, zorder=zorder)
            
            else:  # normal 3d case
                ax.plot(P2[0,:], P2[1,:], P2[2,:], color=color, lw=0.4, zorder=zorder)
        
        
    
    def IECKaimal(self, case, current=False):        # 
        '''Calculates rotor-averaged turbulent wind spectrum based on inputted turbulence intensity or class.'''
        
        #TODO: expand commenting, confirm that Rot is power spectrum, skip V,W calcs if not used

        if current:
            speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
            turbulence = getFromDict(case, 'current_turbulence', shape=0, default=0.0,dtype=str)
        else:
            speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
            turbulence = getFromDict(case, 'turbulence', shape=0, default=0.0,dtype=str)

        # Set inputs (f, V_ref, HH, Class, Categ, TurbMod, R)
        f = self.w / 2 / np.pi    # frequency in Hz
        HH = abs(self.r3[2])     # <<< Temporary absolute value to avoid NaNs with underwater turbines. Eventually need a new function <<<
        R = self.R_rot
        V_ref = speed
        
        ###### Initialize IEC Wind parameters #######
        iec_wind = pyIECWind_extreme()
        iec_wind.z_hub = HH

        # Turbulence can be either a string (IB) or a float (turbulence intensity, 0.1)
        # If a TI is provided, the class defaults to I
        if isinstance(turbulence,str):
            # If a string, the options are I, II, III, IV
            Class = ''
            for char in turbulence:
                if char == 'I' or char == 'V':
                    Class += char
                else:
                    break
            
            if not Class:
                Class = 'I'
                try:
                    turbulence = float(turbulence)
                except:
                    raise Exception(f"Turbulence class must start with I, II, III, or IV: case['turbulence'] = {turbulence}")
            else:
                Categ = char
                iec_wind.Turbulence_Class = Categ

                try:
                    TurbMod = turbulence.split('_')[1]
                except:
                    raise Exception(f"Error reading the turbulence model: {turbulence}")

            iec_wind.Turbine_Class = Class
        
        # set things up (use default values if not specified in the above)
        iec_wind.setup()
        
        # Can set iec_wind.I_ref here if wanted, NTM used then
        if isinstance(turbulence,int):
            turbulence = float(turbulence)
        if isinstance(turbulence,float):
            iec_wind.I_ref = turbulence    # this overwrites the value set in setup method
            TurbMod = 'NTM'

        # Compute wind turbulence standard deviation (invariant with height)
        if TurbMod == 'NTM':
            sigma_1 = iec_wind.NTM(V_ref)
        elif TurbMod == 'ETM':
            sigma_1 = iec_wind.ETM(V_ref)
        elif TurbMod == 'EWM':
            sigma_1 = iec_wind.EWM(V_ref)[0]
        else:
            raise Exception("Wind model must be either NTM, ETM, or EWM. While you wrote " + TurbMod)

        # Compute turbulence scale parameter Annex C3 of IEC 61400-1-2019
        # Longitudinal
        if HH <= 60:
            L_1 = .7 * HH
        else:
            L_1 = 42.
        sigma_u = sigma_1
        L_u = 8.1 * L_1
        # Lateral
        sigma_v =  0.8 * sigma_1
        L_v = 2.7 * L_1 
        # Upward
        sigma_w =  0.5 * sigma_1
        L_w = 0.66 * L_1 

        U = (4*L_u/V_ref)*sigma_u**2/((1+6*f*L_u/V_ref)**(5./3.))
        V = (4*L_v/V_ref)*sigma_v**2/((1+6*f*L_v/V_ref)**(5./3.))
        W = (4*L_w/V_ref)*sigma_w**2/((1+6*f*L_w/V_ref)**(5./3.))

        kappa = 12 * np.sqrt((f/V_ref)**2 + (0.12 / L_u)**2)

        Rot = (2*U / (R * kappa)**3) * \
            (modstruve(1,2*R*kappa) - iv(1,2*R*kappa) - 2/np.pi + \
                R*kappa * (-2 * modstruve(-2,2*R*kappa) + 2 * iv(2,2*R*kappa) + 1) )

        # set NaNs to 0
        Rot[np.isnan(Rot)] = 0

        return U, V, W, Rot
    

    def baldeGeo2Mesh(self, 
                      geometry_table=np.ndarray, 
                      pitch=0., # deg
                      constraint=0.1,
                      mesh_size_max=1,
                      shroud_radius=3, 
                      show=False,
                      mesh=False, 
                      all=False, 
                      save=False,
                      vtk_show=False, 
                      meshFile=os.path.join(raft_dir, "blade_mesh/blade.msh")
                      ):
        '''A function to create RAFT BladeMembers based on rotor blades (Yang currently used). 
        slightly different from the original version bladeGeometry2Member

        Method to create members for each "node" that is specified in turbine['blade']['geometry']
        To be used for added mass and buoyancy calculations of underwater turbines'''

        from raft.helpers import inerpStr
        import gmsh
        gmsh.initialize()

        if geometry_table.shape[1] == 6:
            pitch_axis = geometry_table[:,5]
        else:
            pitch_axis = np.ones_like(geometry_table[:, 0]) * 0.25
        

        af_name = [af.AFName for af in self.ccblade.af]
        af_list = inerpStr(geometry_table[:, 0], self.ccblade.r, af_name)

        curve_loop = []

        for i in range(len(geometry_table[:, 0])):
                  
            file = os.path.join(raft_dir, f"designs/airfoil/coordinate/{af_list[i]}.txt")
            coord = readCoordinate(file)

            if coord[0, 1] == coord[-1, 1]:
                print(f"Warnig: y coordinate of coord[0, 1] and coord[-1, 1] should not be the same, both will be deleted")
                coord = coord[1:-1]

            rpts = []

            n_p = coord.shape[0]  # number of points in airfoil coordinate file
            r_b = np.array([geometry_table[i, 3], geometry_table[i, 4], geometry_table[i, 0]])     # airfoil center in coordinates in blade-aligned coordinate system

            R = rotationMatrix(np.deg2rad(self.azimuths[0]), -np.deg2rad(self.precone), -np.deg2rad(geometry_table[i, 2]-pitch))    # R_z(-twist-pitch) -> R_y(-cone) -> R_x(azimuth) 
            for j in range(n_p):

                x = coord[j, 1] * geometry_table[i, 1] + r_b[0]    # x coordinate in blade-aligned coordinate system
                y = (coord[j, 0]-pitch_axis[i]) * geometry_table[i, 1] + r_b[1]    # y coordinate in blade-aligned coordinate system
                z = r_b[2]                                          # z coordinate in blade-aligned coordinate system

                rpts_rel = np.matmul(self.R_q, np.matmul(R, np.array([x,y,z])))
                rpts.append(gmsh.model.occ.addPoint(rpts_rel[0]+self.r3[0], 
                                                    rpts_rel[1]+self.r3[1], 
                                                    rpts_rel[2]+self.r3[2], 
                                                    geometry_table[i, 1]*constraint))
            
            # rpts.append(rpts[0])
            curv = []
            
            curv.append(gmsh.model.occ.addBSpline(rpts[0: n_p // 2+1]))
            curv.append(gmsh.model.occ.addBSpline(rpts[n_p // 2:]))
            curv.append(gmsh.model.occ.addLine(rpts[-1], rpts[0]))
            curve_loop.append(gmsh.model.occ.addCurveLoop(curv))
        
        # gmsh.model.occ.synchronize()
        # gmsh.model.occ.addThruSections(curv[:6], 1, parametrization="IsoParametric", continuity = "G1" )
        # gmsh.model.occ.addThruSections(curv[5:7], 2, continuity = "G1" )
        # gmsh.model.occ.addThruSections(curv[6:], 3)
        # gmsh.option.setNumber("Mesh.MeshSizeFromCurvatureIsotropic", 1)
        # gmsh.model.occ.synchronize()
        # gmsh.fltk.run()
        gmsh.model.occ.addThruSections(curve_loop, continuity="C2")
        gmsh.model.occ.synchronize()
        # gmsh.fltk.run()
        
        gmsh.option.setNumber("Geometry.NumSubEdges", 100)
        # for iter in range(1, len(curve_loop)):
        #     gmsh.model.occ.addThruSections([iter, iter+1], iter)

        # gmsh.model.occ.fuse([(3,1)], [(3, i) for i in range(2, len(curve_loop))])
        # gmsh.model.occ.synchronize()
        # gmsh.fltk.run()

        if all is True:
            tmp1 = gmsh.model.occ.copy([(3,1)])
            tmp2 = gmsh.model.occ.copy([(3,1)])
            gmsh.model.occ.rotate(tmp1, self.r3[0], self.r3[1], self.r3[2], self.q[0], self.q[1], self.q[2], np.deg2rad(self.azimuths[1]))
            gmsh.model.occ.rotate(tmp2, self.r3[0], self.r3[1], self.r3[2], self.q[0], self.q[1], self.q[2], np.deg2rad(self.azimuths[2]))
            
            gmsh.model.occ.addSphere(self.r3[0],
                                     self.r3[1], 
                                     self.r3[2], 
                                     shroud_radius)
            
            gmsh.model.occ.synchronize()
            gmsh.model.occ.mesh.setSize(gmsh.model.getBoundary([(3,4)], recursive=True), shroud_radius*1.5*constraint)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
            gmsh.model.occ.fuse([(3,1),(3,2),(3,3)], [(3, 4)])

        gmsh.model.occ.synchronize()
        surf_list = gmsh.model.getEntities(2)
        surf = [i[1] for i in surf_list]
        
        if all is True:
            gmsh.model.addPhysicalGroup(2, surf, 1, "rotor")
        
        else:
            gmsh.model.addPhysicalGroup(2, surf, 1, "blade")
        
        if mesh is True:
            gmsh.option.setNumber('Mesh.Smoothing', 5)
            gmsh.option.setNumber('Mesh.SmoothNormals', 1)
            # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 5)
            gmsh.model.mesh.generate(2)
        
        if show is True:
            gmsh.fltk.run()

        if vtk_show is True:
            
            nodes, coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)
            elementType, elements, elementNodes = gmsh.model.mesh.getElements()

            nVertices = len(nodes)
            vertices = np.array(coords).reshape(nVertices, 3)

            if len(elementType[1:-1]) == 2: # triangular and quadature faces
                nTriFaces    = elements[1].shape[0]
                nQuadFaces   = elements[2].shape[0]

                faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
                triFaces = np.hstack((faces_, faces_[:, [0]]))

                quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

                faces = np.asarray(np.vstack((triFaces, quadFaces)))

            elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
                nFaces = elements[1].shape[0]

                faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
                faces = np.hstack((faces_, faces_[:, [0]]))

            elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
                nFaces = elements[1].shape[0]

                faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

            from meshmagick.mesh import Mesh
            from meshmagick.mmio import write_MAR, write_STL, write_VTK

            vtk_mesh = Mesh(vertices ,faces, name="rotor")
            vtk_mesh.show()
            
            if save:
                write_MAR(str(meshFile)[:-3]+'dat', vertices, faces)
                write_STL(str(meshFile)[:-3]+'stl', vertices, faces)
                write_VTK(str(meshFile)[:-3]+'vtk', vertices, faces)
        
        if save:
            gmsh.option.setNumber('Mesh.Format', 1)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.SaveParametric", 0)
            gmsh.write(meshFile)

        gmsh.finalize()



    def bladeMesh(self,
                  pitch = 0.,
                  constraint=0.1,
                  mesh_size_max=1, 
                  shroud_length=6, 
                  shroud_radius=3, 
                  show=False,
                  mesh=False, 
                  all=False, 
                  save=False,
                  vtk_show=False, 
                  meshFile=os.path.join(raft_dir, "blade_mesh/blade.msh")):
        
        '''use gmsh to generate blade mesh of single/each blade of the whole tuebine, hub is currently ignored.
           must be called after setPosition() and setYaw()
        
        Parameters
        ----------        
        show: bool
            If true, produces a 2d plot on the axes defined by Xuvec and Yuvec. 
            Otherwise produces a 3d plot (default).
        : bool
            If true, draw circle of rotor circumference.
        '''
        import gmsh
        
        from raft.helpers import getCone

        r_blade        = np.concatenate(([self.ccblade.Rhub], self.ccblade.r, [self.ccblade.Rtip]))
        chord_blade    = np.concatenate(([self.ccblade.chord[0]], self.ccblade.chord, [self.ccblade.chord[-1]]))
        theta_blade    = np.concatenate(([self.ccblade.theta[0]], self.ccblade.theta, [self.ccblade.theta[-1]]))
        precurve_blade = np.concatenate(([0], self.ccblade.precurve, [self.ccblade.precurveTip]))
        presweep_blade = np.concatenate(([0], self.ccblade.presweep, [self.ccblade.presweepTip]))                                              # not an array
        pitch_axis     = np.concatenate(([self.pitch_axis[0]], self.pitch_axis, [self.pitch_axis[-1]]))

        af_list = self.ccblade.af.copy()
        af_list.insert(0, self.ccblade.af[0])
        af_list.append(self.ccblade.af[-1])

        gmsh.initialize()

        curv = []
        for i, af in enumerate(self.ccblade.af):

            file = os.path.join(raft_dir, f"designs/airfoil/coordinate/{af.AFName}.txt")
            coord = readCoordinate(file)

            n_p = coord.shape[0]    # number of points in airfoil coordinate file
            r_b = np.array([precurve_blade[i], presweep_blade[i], r_blade[i]])     # airfoil center in coordinates in blade-aligned coordinate system

            pts = []
            for j in range (n_p):

                x = coord[j, 1] * chord_blade[i] + r_b[0]    # x coordinate in blade-aligned coordinate system
                y = (coord[j, 0]-pitch_axis[i]) * chord_blade[i] + r_b[1]    # y coordinate in blade-aligned coordinate system
                z = r_b[2]                                          # z coordinate in blade-aligned coordinate system

                r0 = np.array([x,y,z])   # coordinates in blade-aligned coordinate system
                R = rotationMatrix(np.deg2rad(self.azimuths[0]), -np.deg2rad(self.precone), -theta_blade[i]-pitch)    # R_z(-twist-pitch) * R_y(-cone) * R_x(azimuth) 
                                                                                    # coordinates in azimuth-aligned coordinate system
                
                r1 = np.matmul(self.R_q, np.matmul(R, r0))                         # coordinates in absolute coordinate system
                r  = np.array([r1[0]+self.r3[0], r1[1]+self.r3[1], r1[2]+self.r3[2]])

                if j == 0:
                    pts.append(gmsh.model.occ.addPoint(r[0], r[1], r[2], chord_blade[i]*constraint*0.8))
                elif j == n_p // 2 or j == n_p // 2+1:
                    pts.append(gmsh.model.occ.addPoint(r[0], r[1], r[2], chord_blade[i]*constraint*0.8))
                else:
                    pts.append(gmsh.model.occ.addPoint(r[0], r[1], r[2], chord_blade[i]*constraint))

            pts.append(pts[0])

            curv.append(gmsh.model.occ.addBSpline(pts))
            gmsh.model.occ.addCurveLoop([curv[i]])

        for iter in range(len(curv) - 1):
            gmsh.model.occ.addThruSections([curv[iter], curv[iter+1]], iter+1, maxDegree=2)

        gmsh.model.occ.fuse([(3,1)], [(3, i) for i in range(2, len(curv))])
        gmsh.model.occ.synchronize()


        if all is True:

            tmp1 = gmsh.model.occ.copy([(3,1)])
            tmp2 = gmsh.model.occ.copy([(3,1)])
            gmsh.model.occ.rotate(tmp1, self.r3[0], self.r3[1], self.r3[2], self.q[0], self.q[1], self.q[2], np.deg2rad(self.azimuths[1]))
            gmsh.model.occ.rotate(tmp2, self.r3[0], self.r3[1], self.r3[2], self.q[0], self.q[1], self.q[2], np.deg2rad(self.azimuths[2]))

            gmsh.model.occ.addCylinder(self.r3[0]+self.q[0]*-0.5*shroud_length,
                                       self.r3[1]+self.q[1]*-0.5*shroud_length, 
                                       self.r3[2]+self.q[2]*-0.5*shroud_length, 
                                       self.q[0]*shroud_length, 
                                       self.q[1]*shroud_length, 
                                       self.q[2]*shroud_length, 
                                       shroud_radius)

            gmsh.model.occ.synchronize()
            gmsh.model.occ.mesh.setSize(gmsh.model.getBoundary([(3,4)], recursive=True), shroud_radius*constraint)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
            gmsh.model.occ.fuse([(3,1),(3,2),(3,3)], [(3, 4)])

        # def test():

        #     pts_test = []
        #     pts_test.append(gmsh.model.occ.addPoint(self.r3[0], self.r3[1], self.r3[2]))

        #     r_b = np.array([-self.ccblade.precurve[-1], self.ccblade.presweep[-1], self.ccblade.r[-1]+self.Rhub])
            
        #     R = rotationMatrix(np.deg2rad(self.azimuths[0]), -cone[-1], -self.ccblade.theta[-1])  
        #     r_blade_0 = np.matmul(self.R_q, np.matmul(R, r_b)) 
        #     r_blade_0 = np.array([r_blade_0[0]+self.r3[0], r_blade_0[1]+self.r3[1], r_blade_0[2]+self.r3[2]])

        #     R = rotationMatrix(np.deg2rad(self.azimuths[1]), -cone[-1], -self.ccblade.theta[-1])
        #     r_blade_1 = np.matmul(self.R_q, np.matmul(R, r_b))
        #     r_blade_1 = np.array([r_blade_1[0]+self.r3[0], r_blade_1[1]+self.r3[1], r_blade_1[2]+self.r3[2]])

        #     R = rotationMatrix(np.deg2rad(self.azimuths[2]), -cone[-1], -self.ccblade.theta[-1])
        #     r_blade_2 = np.matmul(self.R_q, np.matmul(R, r_b))
        #     r_blade_2 = np.array([r_blade_2[0]+self.r3[0], r_blade_2[1]+self.r3[1], r_blade_2[2]+self.r3[2]])

        #     q_blade = self.q*63

        #     pts_test.append(gmsh.model.occ.addPoint(r_blade_0[0], r_blade_0[1], r_blade_0[2]))
        #     pts_test.append(gmsh.model.occ.addPoint(r_blade_1[0], r_blade_1[1], r_blade_1[2]))
        #     pts_test.append(gmsh.model.occ.addPoint(r_blade_2[0], r_blade_2[1], r_blade_2[2]))
        #     pts_test.append(gmsh.model.occ.addPoint(q_blade[0]+self.r3[0], q_blade[1]+self.r3[1], q_blade[2]+self.r3[2]))

        #     gmsh.model.occ.addLine(pts_test[0], pts_test[1])
        #     gmsh.model.occ.addLine(pts_test[0], pts_test[2])
        #     gmsh.model.occ.addLine(pts_test[0], pts_test[3])
        #     gmsh.model.occ.addLine(pts_test[0], pts_test[4])

        # test()

        gmsh.model.occ.synchronize()
        surf_list = gmsh.model.getEntities(2)
        surf = [i[1] for i in surf_list]

        if all is True:
            gmsh.model.addPhysicalGroup(2, surf, 2, "rotor")
        else:
            gmsh.model.addPhysicalGroup(2, surf, 2, "blade")
            
        if mesh is True:
            gmsh.option.setNumber('Mesh.Smoothing', 5)
            gmsh.option.setNumber('Mesh.SmoothNormals', 1)
            gmsh.model.mesh.generate(2)
            # gmsh.model.mesh.optimize()
        
        if vtk_show is True:
            
            nodes, coords, _ = gmsh.model.mesh.getNodes()
            nodes, coords, _ = gmsh.model.mesh.getNodes()
            elementType, elements, elementNodes = gmsh.model.mesh.getElements()

            nVertices = len(nodes)
            vertices = np.array(coords).reshape(nVertices, 3)

            if len(elementType[1:-1]) == 2: # triangular and quadature faces
                nTriFaces    = elements[1].shape[0]
                nQuadFaces   = elements[2].shape[0]

                faces_ = np.array(elementNodes[1]-1).reshape(nTriFaces ,3)
                triFaces = np.hstack((faces_, faces_[:, [0]]))

                quadFaces = np.array(elementNodes[2]-1).reshape(nQuadFaces ,4)

                faces = np.asarray(np.vstack((triFaces, quadFaces)))

            elif len(elementType[1:-1]) == 1 and elementType[1] == 2: # all triangular faces
                nFaces = elements[1].shape[0]

                faces_ = np.array(elementNodes[1]-1).reshape(nFaces ,3)
                faces = np.hstack((faces_, faces_[:, [0]]))

            elif len(elementType[1:-1]) == 1 and elementType[1] == 3: # all rectangular faces
                nFaces = elements[1].shape[0]

                faces = np.array(elementNodes[1]-1).reshape(nFaces ,4)

            from meshmagick.mesh import Mesh

            vtk_mesh = Mesh(vertices ,faces, name="rotor")
            vtk_mesh.show()

        if show is True:
            gmsh.fltk.run()

        if save is True:
            gmsh.option.setNumber('Mesh.Format', 1)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.SaveParametric", 0)

            gmsh.write(meshFile)

        gmsh.finalize()

if __name__=='__main__':
    fname_design = os.path.join(raft_dir,'designs/VolturnUS-S.yaml')

    # open the design YAML file and parse it into a dictionary for passing to raft
    with open(fname_design) as file:
        design = yaml.load(file, Loader=yaml.FullLoader)

    # transfer some dictionary contents that would normally be done higher up in RAFT
    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']
    
    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0


    UU = np.arange(6,17,2)           # wind speeds
    ws = np.arange(0.01,6.0,0.01) # frequencies (rad/s)

    # make Rotor object
    rotor = Rotor(design['turbine'], ws, 0)    
    
    
    

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    
    cmapper = cm.get_cmap('inferno_r')

    fig,ax = plt.subplots(2,2,sharex=True, figsize=(10,4.8))
    
    # loop through each case    
    for i_case in range(len(UU)):
    
        # manually set the case
        case = dict(wind_speed    = UU[i_case], 
                    wind_heading  = 0, 
                    turbulence    = 'IB_NTM', 
                    turbine_status= 'operating', 
                    yaw_misalign  = 0, 
                    wave_spectrum = 'JONSWAP', 
                    wave_period   = 8, 
                    wave_height   = 2, 
                    wave_heading  = 0 )
        
    
        print(f"  Running case {i_case}")
        
        f_aero0, f_aero, a_aero, b_aero = rotor.calcAero(case)


        rgba = cmapper((i_case+1)/8)

        ax[0,0].plot(ws/2.0/np.pi, a_aero[4, 4, :]     , color=rgba, label=f"U = {UU[i_case]:2.0f} m/s")
        ax[1,0].plot(ws/2.0/np.pi, b_aero[4, 4, :]     , color=rgba, label=f"U = {UU[i_case]:2.0f} m/s")
        ax[0,1].plot(ws/2.0/np.pi, np.real(rotor.c_exc), color=rgba)
        ax[0,1].plot(ws/2.0/np.pi, np.imag(rotor.c_exc), color=rgba, ls=":") 
        ax[1,1].plot(ws/2.0/np.pi, rotor.V_w           , color=rgba, label=f"U = {UU[i_case]:2.0f} m/s")

    
    ax[0,1].plot([],[], color=[0.5,0.5,0.5],        label='real')
    ax[0,1].plot([],[], color=[0.5,0.5,0.5], ls=":",label='imaginary')

    ax[0,0].set_ylabel(r"$a_{aero}(\omega)$ (kg)")  
    ax[1,0].set_ylabel(r"$b_{aero}(\omega)$ (Ns/m)") 
    ax[0,1].set_ylabel(r"$H_{Uf}(\omega)$ (Ns/m)") 
    ax[1,1].set_ylabel(r"$U(\omega)$ (m/s)") 
    ax[1,0].set_xlabel(r"frequency (Hz)") 
    ax[1,1].set_xlabel(r"frequency (Hz)") 
    ax[1,1].set_xlim([0,0.2]) 
    
    # force to use exponent y axis labeling
    ax[0,0].ticklabel_format(axis='y', scilimits=[-3, 3])
    ax[1,0].ticklabel_format(axis='y', scilimits=[-3, 3])
    ax[0,1].ticklabel_format(axis='y', scilimits=[-3, 3])
    ax[1,1].ticklabel_format(axis='y', scilimits=[-3, 3])
    
    ax[1,0].set_xticks(np.arange(0, 0.21,0.05))
    
    ax[0,0].grid()
    ax[1,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    
    ax[0,0].legend()
    ax[0,1].legend()
    fig.align_ylabels()
    
    fig.tight_layout()
    fig.savefig("control.png", dpi=200)
    plt.show()