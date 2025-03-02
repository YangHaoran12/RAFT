from raft.raft_fowt import FOWT
from raft.raft_member import TowerMember
import numpy as np
from raft.helpers import transformForce

def calcStaticTwrBendingMoment(fowt:FOWT, case, cal_stress=True):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)
            calc_stress (Boolean): wheather to calculate stress causing by twr bending moment at touwer base

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     

        Mbase_list = np.zeros([2, fowt.nrotors])
        stress_list = np.zeros([2, fowt.nrotors])

        fowt.setPosition([0,0,0,0,0,0])
        # fowt.calcTurbineConstants(case)
        fowt.calcTowerAeroLoads(case)

        m_turbine = np.zeros(fowt.ntowers)
        rCG_turbine = np.zeros([fowt.nrotors, 3])
        rBase = np.zeros([fowt.ntowers, 3])
        rArm = np.zeros([fowt.ntowers, 3])
        f_aero0 = np.zeros([fowt.ntowers, 6])

        for ir, rot in enumerate(fowt.rotorList):
            # mass and moment arm >>> should three-dimensionalize <<<

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)
            mass, center, m_shell, mfill, pfill = mem.getInertia(rPRP=fowt.r6[:3]) 
            m_turbine[ir] = mass + rot.mRNA  # total masses of each turbine

            rCG_turbine[ir] = (center*mass+ rot.r_CG_rel*rot.mRNA)/m_turbine[ir]    
                                        
            # towerMebmerList = [mem for mem in self.memberList if type(mem) is TowerMember]
            rBase[ir] = mem.rA - fowt.r6[:3]                  # tower base elevation [m]
            rArm[ir] = rCG_turbine[ir] - rBase[ir]        

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)
            
            # use symmetric F_y and M_x to represent aero force from 
            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]

            R_tower = mem.R
            Mbase_avg = (np.cross(rArm[ir], m_turbine[ir]*fowt.g*np.array([0,0,-1]))
                        + transformForce(f_aero0[ir], offset=(rot.r_hub_rel-mem.rA))[3:]
                        + transformForce(fowt.F_tower_aero[:,ir], offset=-rBase[ir])[3:])
            
            # print(f'f_aero0[{ir}] = {f_aero0[ir]}')
            # print(f'rBase[{ir}] = {rBase[ir]}')
            # print(f'F_tower_aero[{ir}] = {fowt.F_tower_aero[:,ir]}')
            # print(f'Mbase_avg{ir}] = {Mbase_avg}')
            Mbase_rel_avg = np.matmul(R_tower, Mbase_avg)

            Mbase_list[:, ir] = Mbase_rel_avg[0:2]

            if cal_stress is True:

                if mem.shape == 'circular':                  
                      stress_list[0, ir] = 4*Mbase_list[0, ir]/(np.pi*mem.d[0]**2*mem.t[0])
                      stress_list[1, ir] = 4*Mbase_list[1, ir]/(np.pi*mem.d[0]**2*mem.t[0])

                elif mem.shape == 'ellipse':
                    #   Ixx = 0.25*np.pi*mem.sl[0,1]*mem.t[0]*mem.sl[0,0]
                    #   Iyy = 0.25*np.pi*mem.sl[0,0]*mem.t[0]*mem.sl[0,1]

                      stress_list[0, ir] = 4*Mbase_list[0, ir]/(np.pi*mem.sl[0,0]*mem.sl[0,1]**2*mem.t[0])
                      stress_list[1, ir] = 4*Mbase_list[1, ir]/(np.pi*mem.sl[0,1]*mem.sl[0,0]**2*mem.t[0])
                else:
                      raise NotImplementedError()
                  

        return Mbase_list, stress_list