from raft.raft_fowt import FOWT
from raft.raft_member import TowerMember
import numpy as np
from raft.helpers import transformForce, translateMatrix6to6DOF
from Pynite import FEModel3D
from Pynite.PhysMember import PhysMember
import matplotlib.pyplot as plt
from matplotlib import cm
    
    

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


def solveTwrFEA(fowt:FOWT, case, plot=True):
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

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)
            
            # use symmetric F_y and M_x to represent aero force from 
            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top = transformForce(f_aero0[ir], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0],offset=(rot.r_CG_rel-mem.rB))
            
            E = 210*1e9        # Modulus of elasticity (Pa)
            G = 80.8*1e9       # Shear modulus of elasticity (Pa)
            nu = 0.3           # Poisson's ratio
            rho = 8500         # Density (kg/m**3)

            beam = FEModel3D()
            beam.add_material('Steel', E, G, nu, rho)
            
            for il in range(len(mem.ls)):
                
                xs, ys, zs = mem.q*mem.ls[il]
                beam.add_node(f'N{il}', xs, ys, zs)
                

            for il in range(len(mem.ls)-1):

                A0, Iy0, Iz0, J0 = section_property(mem.ds[il], mem.ts[il])
                A1, Iy1, Iz1, J1 = section_property(mem.ds[il+1], mem.ts[il+1])
                A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
                # A, Iy, Iz, J = np.array([A0, Iy0, Iz0, J0])
                beam.add_section(f'Section{il}', A, Iy, Iz, J, d=mem.ds[il], t=mem.ts[il])

                beam.add_member(f'M{il}', f'N{il}', f'N{il+1}', 'Steel', f'Section{il}')
                
                # sectional aero loads
                beam.add_member_dist_load(f'M{il}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
                beam.add_member_dist_load(f'M{il}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
                beam.add_member_dist_load(f'M{il}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])


            beam.def_support('N0', True, True, True, True, True, True)
            # beam.def_support(f'N{il+1}', True, True, False, False, False, False)
            # beam.def_support(f'N{11}', True, True, False, False, False, False)
            beam.def_support(f'N{il+1}', False, False, False, False, False, False)
            # beam.def_releases(f'M{il}',0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0)

            beam.add_member_self_weight('FZ', factor=-1)

            beam.add_node_load(f'N{il+1}', 'FX', f_aero_twr_top[0])
            beam.add_node_load(f'N{il+1}', 'FY', f_aero_twr_top[1])
            beam.add_node_load(f'N{il+1}', 'FZ', f_aero_twr_top[2])
            beam.add_node_load(f'N{il+1}', 'MX', f_aero_twr_top[3])
            beam.add_node_load(f'N{il+1}', 'MY', f_aero_twr_top[4])
            beam.add_node_load(f'N{il+1}', 'MZ', f_aero_twr_top[5])

            # M_RNA = np.array([rot.mRNA, rot.mRNA, rot.mRNA,  rot.IxRNA, rot.IrRNA, rot.IrRNA])
            
            M_RNA = np.array([[350000. , 0.        , 0.       , 0.          , 688446.5   , 0.         ],
                              [0.      , 350000.   , 0.       , -688446.5   , 0.         , -144821.25 ],
                              [0.      , 0.        , 350000.  , 0.          , 144821.25  , 0.         ],
                              [0.      , -688446.5 , 0.       , 45054167.38 , 0.         , 1453861.95 ],
                              [688446.5, 0.        , 144821.25, 0.          , 21414090.79, 0.         ],
                              [0.      , -144821.25, 0.       , 1453861.95  , 0.         , 20059923.41]])
            id = np.array([il*6])
        
            fq6, eigen_v = beam.analyze_eigen(id=id, added_mass=M_RNA) # Hz # first 6st mode frequency 
            
            beam.analyze()
            
            misses_y=[]
            misses_z=[]
            
            for i, member in enumerate(beam.members.values()):
                x0, y0 = anaylyze_stress(member)
                misses_y.append(x0)     
                misses_z.append(y0)

            def plot_mod():

                DX_mod = []
                DY_mod = []
                RX_mod = []
                RY_mod = []
                
                for i in range(len(mem.ls)-1):
                     DX_mod.append(eigen_v[6*i,   [0,1,3,4]])
                     DY_mod.append(eigen_v[6*i+1, [0,1,3,4]])
                     RX_mod.append(eigen_v[6*i+3, [0,1,3,4]])
                     RY_mod.append(eigen_v[6*i+4, [0,1,3,4]])

                mod_name = ['side-side I', 'fore-aft I', 'side-side II', 'fore-aft II']

                fig, ax = plt.subplots(4)    
                

                ax[0].plot(mem.ls[1:], DX_mod, label=mod_name, ls='--', linewidth=2)
                ax[0].grid()
                ax[0].set_ylabel('DX')
                ax[0].legend()

                ax[1].plot(mem.ls[1:], DY_mod, label=mod_name, ls='--', linewidth=2)
                ax[1].grid()
                ax[1].set_ylabel('DY')
                ax[1].legend()

                ax[2].plot(mem.ls[1:], RX_mod, label=mod_name, ls='--', linewidth=2)
                ax[2].grid()
                ax[2].set_ylabel('RX')
                ax[2].legend()

                ax[3].plot(mem.ls[1:], RY_mod, label=mod_name, ls='--', linewidth=2)
                ax[3].grid()
                ax[3].set_ylabel('RY')
                ax[3].legend()
                
                return ax
        
            def plot_statics():
                
                My = []
                Mz = []
                Dx = []
                Dy = [] 
                Dz = [] 
                    
                for i, member in enumerate(beam.members.values()):

                    My.append(member.moment('My',0))
                    Mz.append(member.moment('Mz',0))
                    
                    Dx.append(member.deflection('dx', 0))
                    Dy.append(member.deflection('dy', 0))
                    Dz.append(member.deflection('dz', 0))
        
        
                # import matplotlib.pyplot as plt
                # plt.plot(mem.ls[:-1], x[:nn-1])
                # plt.plot(mem.ls[:-1], y[:nn-1])
                fig, ax = plt.subplots(3)
                ax[0].plot(mem.ls[1:], misses_y, label='Mises y')
                ax[0].plot(mem.ls[1:], misses_z, label='Mises z')
                ax[0].grid()
                ax[0].legend()
        
                ax[1].plot(mem.ls[1:], My, label='My')
                ax[1].plot(mem.ls[1:], Mz, label='Mz')
                ax[1].grid()
                ax[1].legend()
        
                # ax[2].plot(Dx, label='Dx')
                # ax[2].plot(Dy, label='Dy')
                ax[2].plot(mem.ls[1:], Dz, label='Dz')
                ax[2].grid()
                ax[2].legend()

                print(f'node Dispx of top = {Dx[-1]:.4e}')
                print(f'node Dispy of top = {Dy[-1]:.4e}')
                print(f'node Dispz of top = {Dz[-1]:.4e}')
        
                return ax
            
            if plot is True:
                ax0 = plot_mod()
                ax1 = plot_statics()
                plt.show()
           

        return np.max(misses_y), np.max(misses_z), fq6


def solveTwrCombinationFEA(fowt:FOWT, case, plot=True):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     

        Mbase_list = np.zeros([2, fowt.nrotors])
        stress_list = np.zeros([2, fowt.nrotors])

        fowt.setPosition([0,0,0,0,0,0])
        # fowt.calcTurbineConstants(case)
        fowt.calcTowerAeroLoads(case)

        # m_turbine = np.zeros(fowt.ntowers)
        # rCG_turbine = np.zeros([fowt.nrotors, 3])
        # rBase = np.zeros([fowt.ntowers, 3])
        # rArm = np.zeros([fowt.ntowers, 3])
        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)

        beam = FEModel3D()
        beam.add_material('Steel', E, G, nu, rho)

        for ir, rot in enumerate(fowt.rotorList):
            # mass and moment arm >>> should three-dimensionalize <<<

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG_rel-mem.rB))
            
            for il in range(len(mem.ls)):
                
                nNode = len(mem.ls)
                nn = ir*nNode # ir = 0, nn = 0; ir = 1, nn = nNode
                
                if ir == 0:
                    xs, ys, zs = mem.rA + mem.q*mem.ls[il]
                    beam.add_node(f'N{il}', xs, ys, zs)
                elif ir > 0 and il == 0:
                    pass
                else:
                    xs, ys, zs = mem.rA + mem.q*mem.ls[il]
                    beam.add_node(f'N{il+nn-1}', xs, ys, zs)

            for il in range(len(mem.ls)-1):
                
                nNode = len(mem.ls)
                nn = ir*nNode
                
                # A0, Iy0, Iz0, J0 = section_property(mem.ds[il], mem.ts[il], type='ellipse')
                # A1, Iy1, Iz1, J1 = section_property(mem.ds[il+1], mem.ts[il+1], type='ellipse')
                A0, Iy0, Iz0, J0 = section_property(mem.ds[il], mem.ts[il])
                A1, Iy1, Iz1, J1 = section_property(mem.ds[il+1], mem.ts[il+1])
                A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
                
                if ir == 0:
                    beam.add_section(f'Section{il}', A, Iy, Iz, J, d=mem.ds[il], t=mem.ts[il])

                if ir == 1 and il == 0:
                    beam.add_member(f'M{il+nn-1}', f'N{0}', f'N{il+nn}', 'Steel', f'Section{0}')
                    
                    # sectional aero loads
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])
                    
                    #try
                    # beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])
                elif ir == 0:
                    beam.add_member(f'M{il}', f'N{il}', f'N{il+1}', 'Steel', f'Section{il}')
                    
                    # sectional aero loads
                    beam.add_member_dist_load(f'M{il}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
                    beam.add_member_dist_load(f'M{il}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
                    beam.add_member_dist_load(f'M{il}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])

                    #try
                    # beam.add_member_dist_load(f'M{il}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])
                elif ir == 1 and il > 0:
                    beam.add_member(f'M{il+nn-1}', f'N{il+nn-1}', f'N{il+nn}', 'Steel', f'Section{il}')
                
                    # sectional aero loads
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
                    beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])

                    #try
                    # beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])

        d_ten = 0.25
        A_ten = 0.25*np.pi*d_ten**2
        beam.add_material('Tension', 2.1e11, 1e-12, 0.3, 1e-12)
        beam.add_section(f'Section{il+nn+1}', A_ten, 0.0, 0.0, 0.0)

        beam.add_member(f'M{il+nn}', f'N{int((nn-1)/2)}', f'N{int((3*nn-3)/2)}', 'Tension', f'Section{il+nn+1}', tension_only=True)
        beam.add_member(f'M{il+nn+1}', f'N{int(nn-1)}',   f'N{int(2*nn-2)}'  , 'Tension', f'Section{il+nn+1}', tension_only=True)
            
        beam.def_support('N0', True, True, True, True, True, True)
        beam.def_support(f'N{nn-1}', False, False, False, False, False, False)
        beam.def_support(f'N{2*nn-2}', False, False, False, False, False, False)

        beam.def_releases(f'M{il+nn}',   0,0,0,1,1,1,0,0,0,1,1,1)
        beam.def_releases(f'M{il+nn+1}', 0,0,0,1,1,1,0,0,0,1,1,1)

        
        beam.add_member_self_weight('FZ', factor=-1)

        beam.add_node_load(f'N{nn-1}', 'FX', f_aero_twr_top[0][0])
        beam.add_node_load(f'N{nn-1}', 'FY', f_aero_twr_top[0][1])
        beam.add_node_load(f'N{nn-1}', 'FZ', f_aero_twr_top[0][2])
        beam.add_node_load(f'N{nn-1}', 'MX', f_aero_twr_top[0][3])
        beam.add_node_load(f'N{nn-1}', 'MY', f_aero_twr_top[0][4])
        beam.add_node_load(f'N{nn-1}', 'MZ', f_aero_twr_top[0][5])

        beam.add_node_load(f'N{2*nn-2}', 'FX', f_aero_twr_top[1][0])
        beam.add_node_load(f'N{2*nn-2}', 'FY', f_aero_twr_top[1][1])
        beam.add_node_load(f'N{2*nn-2}', 'FZ', f_aero_twr_top[1][2])
        beam.add_node_load(f'N{2*nn-2}', 'MX', f_aero_twr_top[1][3])
        beam.add_node_load(f'N{2*nn-2}', 'MY', f_aero_twr_top[1][4])
        beam.add_node_load(f'N{2*nn-2}', 'MZ', f_aero_twr_top[1][5])

        # beam.analyze()
        # M_RNA = np.array([rot.mRNA, rot.mRNA, rot.mRNA,  rot.IxRNA, rot.IrRNA, rot.IrRNA])
        # M_RNA = translateMatrix6to6DOF(M_RNA, [-0.27,0,0])
        
        # id1 = np.arange((nn-1)*6, nn*6)
        # id2 = np.arange((2*nn-3)*6, (2*nn-2)*6)
        # Twr_top_indices = np.hstack((id1, id2))
        # added_mass_RNA = np.hstack((M_RNA, M_RNA))
        
        id = [(nn-2)*6, (2*nn-3)*6]    
        M_RNA = np.array([[350000. , 0.        , 0.       , 0.          , 688446.5   , 0.         ],
                          [0.      , 350000.   , 0.       , -688446.5   , 0.         , -144821.25 ],
                          [0.      , 0.        , 350000.  , 0.          , 144821.25  , 0.         ],
                          [0.      , -688446.5 , 0.       , 45054167.38 , 0.         , 1453861.95 ],
                          [688446.5, 0.        , 144821.25, 0.          , 21414090.79, 0.         ],
                          [0.      , -144821.25, 0.       , 1453861.95  , 0.         , 20059923.41]])
        # M_RNA = np.zeros([6,6])
        fq6, eigen_v = beam.analyze_eigen(id=id, added_mass=M_RNA) # Hz # first 6st mode frequency 
        eigen_v = np.vstack([np.zeros([6,6]), eigen_v])    
        
        # static 
        beam.analyze()

        # cable stress
        cable_ele = beam.members[f'M{il+nn+1}']
        F_cable = cable_ele.axial(0)
        cable_stress = np.abs(F_cable/A_ten)

        misses_y=[]
        misses_z=[]

        for i, member in enumerate(beam.members.values()):
                # if i < 2*nn-2:
                if i < nn-1:
                    x0, y0 = anaylyze_stress(member)
                    misses_y.append(x0)
                    misses_z.append(y0)

                elif i >= nn-1 and i < 2*nn-2:
                
                    x0, y0 = anaylyze_stress(member, factor=-1)
                    misses_y.append(x0)
                    misses_z.append(y0)

                else:
                    break

        def plot_mod():
            
            nf = eigen_v.shape[1]
            DX_mod = np.zeros((len(mem.ls)-1, nf))
            DY_mod = np.zeros((len(mem.ls)-1, nf))
            DZ_mod = np.zeros((len(mem.ls)-1, nf))
            RX_mod = np.zeros((len(mem.ls)-1, nf))
            RY_mod = np.zeros((len(mem.ls)-1, nf))
            RZ_mod = np.zeros((len(mem.ls)-1, nf))

            DX_mod_1 = np.zeros((len(mem.ls)-1, nf))
            DY_mod_1 = np.zeros((len(mem.ls)-1, nf))
            DZ_mod_1 = np.zeros((len(mem.ls)-1, nf))
            RX_mod_1 = np.zeros((len(mem.ls)-1, nf))
            RY_mod_1 = np.zeros((len(mem.ls)-1, nf))
            RZ_mod_1 = np.zeros((len(mem.ls)-1, nf))
            
            for i in range(0, len(mem.ls)-1):
                 for fs in range(nf):  

                    # d = beam.members[f'M{i}'].T() @ eigen_v[6*i:6*i+12, fs]
                    d = eigen_v[6*i:6*i+12, fs]
                    
                    DX_mod[i][fs] = d[0]
                    DY_mod[i][fs] = d[1]
                    DZ_mod[i][fs] = d[2]
                    RX_mod[i][fs] = d[3]
                    RY_mod[i][fs] = d[4]
                    RZ_mod[i][fs] = d[5]    

            for i in range(0, len(mem.ls)-1):
                 for fs in range(nf):  
                    
                    # d = beam.members[f'M{i+nn-1}'].T() @ eigen_v[6*(i+nn-2):6*(i+nn-2)+12, fs]
                    if i == 0:
                        d = np.zeros(6)
                    else:
                        # d = -beam.members[f'M{i+nn-1}'].T().T @ eigen_v[6*(i+nn-1):6*(i+nn-1)+12, fs]
                        d = eigen_v[6*(i+nn-1):6*(i+nn-1)+12, fs]
                    
                    DX_mod_1[i][fs] = d[0]
                    DY_mod_1[i][fs] = d[1]
                    DZ_mod_1[i][fs] = d[2]
                    RX_mod_1[i][fs] = d[3]
                    RY_mod_1[i][fs] = d[4]
                    RZ_mod_1[i][fs] = d[5]   
            
            mod_name = ['1st', '2nd', '3rd', '4th','5th','6th']
            fig, ax = plt.subplots(6,2)
            for iax in range(6):
                 ax[iax,0].sharey(ax[iax,1]) 
            
            ax[0,0].plot(mem.ls[1:], DX_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[0,0].grid()
            ax[0,0].set_ylabel('Dx')
            ax[0,0].legend()
            
            ax[1,0].plot(mem.ls[1:], DY_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[1,0].grid()
            ax[1,0].set_ylabel('Dy')
            ax[1,0].legend()
            
            ax[2,0].plot(mem.ls[1:], DZ_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[2,0].grid()
            ax[2,0].set_ylabel('Dz')
            ax[2,0].legend()
            
            ax[3,0].plot(mem.ls[1:], RX_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[3,0].grid()
            ax[3,0].set_ylabel('Rx')
            ax[3,0].legend()

            ax[4,0].plot(mem.ls[1:], RY_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[4,0].grid()
            ax[4,0].set_ylabel('Ry')
            ax[4,0].legend()

            ax[5,0].plot(mem.ls[1:], RZ_mod[:,:], label=mod_name, ls='--', linewidth=2)
            ax[5,0].grid()
            ax[5,0].set_ylabel('Rz')
            ax[5,0].legend()

            ax[0,1].plot(mem.ls[1:], DX_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[0,1].grid()
            ax[0,1].set_ylabel('Dx')
            ax[0,1].legend()
            
            ax[1,1].plot(mem.ls[1:], DY_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[1,1].grid()
            ax[1,1].set_ylabel('Dy')
            ax[1,1].legend()
            
            ax[2,1].plot(mem.ls[1:], DZ_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[2,1].grid()
            ax[2,1].set_ylabel('Dz')
            ax[2,1].legend()
            
            ax[3,1].plot(mem.ls[1:], RX_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[3,1].grid()
            ax[3,1].set_ylabel('Rx')
            ax[3,1].legend()

            ax[4,1].plot(mem.ls[1:], RY_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[4,1].grid()
            ax[4,1].set_ylabel('Ry')
            ax[4,1].legend()

            ax[5,1].plot(mem.ls[1:], RZ_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
            ax[5,1].grid()
            ax[5,1].set_ylabel('Rz')
            ax[5,1].legend()
            
            return ax 

        def plot_static():
            
            
            My = []
            Mz = []
            Dx = []
            Dy = [] 
            Dz = [] 

            for i, member in enumerate(beam.members.values()):
                # if i < 2*nn-2:
                if i < nn-1:

                    My.append(member.moment('My',0))
                    Mz.append(member.moment('Mz',0))

                    Dx.append(member.deflection('dx', 0))
                    Dy.append(member.deflection('dy', 0))
                    Dz.append(member.deflection('dz', 0))

                elif i >= nn-1 and i < 2*nn-2:
                
                    My.append(member.moment('My',0))
                    Mz.append(member.moment('Mz',0))

                    Dx.append(member.deflection('dx', 0))
                    Dy.append(member.deflection('dy', 0))
                    Dz.append(member.deflection('dz', 0))

                else:
                    break
            

            # plt.plot(mem.ls[:-1], x[:nn-1])
            # plt.plot(mem.ls[:-1], y[:nn-1])
            fig, ax = plt.subplots(4,2)
            ax[0,0].plot(mem.ls[:-1], misses_y[:nn-1], label='Mises y')
            ax[0,0].plot(mem.ls[:-1], misses_z[:nn-1], label='Mises z')
            ax[0,0].grid()
            ax[0,0].legend()

            ax[1,0].plot(mem.ls[:-1], My[:nn-1], label='My')
            ax[1,0].plot(mem.ls[:-1], Mz[:nn-1], label='Mz')
            ax[1,0].grid()
            ax[1,0].legend()

            ax[2,0].plot(mem.ls[:-1], Dy[:nn-1], label='Dy')
            ax[2,0].grid()
            ax[2,0].legend()

            ax[3,0].plot(mem.ls[:-1], Dz[:nn-1], label='Dz')
            ax[3,0].grid()
            ax[3,0].legend()

            ax[0,1].plot(mem.ls[:-1], misses_y[nn-1:], label='Mises y')
            ax[0,1].plot(mem.ls[:-1], misses_z[nn-1:], label='Mises z')
            ax[0,1].grid()
            ax[0,1].legend()

            ax[1,1].plot(mem.ls[:-1], My[nn-1:], label='My')
            ax[1,1].plot(mem.ls[:-1], Mz[nn-1:], label='Mz')
            ax[1,1].grid()
            ax[1,1].legend()

            ax[2,1].plot(mem.ls[:-1], Dy[nn-1:], label='Dy')
            ax[2,1].grid()
            ax[2,1].legend()

            ax[3,1].plot(mem.ls[:-1], Dz[nn-1:], label='Dz')
            ax[3,1].grid()
            ax[3,1].legend()
            
            return ax

        if plot is True:
            ax0 = plot_mod()
            ax1 = plot_static()
            plt.show()

        return np.max(misses_y), np.max(misses_z), cable_stress, fq6

def PostProcessMBDyn(fowt:FOWT, case, plot=True):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     

        # Mbase_list = np.zeros([2, fowt.nrotors])
        # stress_list = np.zeros([2, fowt.nrotors])

        fowt.setPosition([0,0,0,0,0,0])
        # fowt.calcTurbineConstants(case)
        fowt.calcTowerAeroLoads(case)

        # m_turbine = np.zeros(fowt.ntowers)
        # rCG_turbine = np.zeros([fowt.nrotors, 3])
        # rBase = np.zeros([fowt.ntowers, 3])
        # rArm = np.zeros([fowt.ntowers, 3])
        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)
        
        for ir, rot in enumerate(fowt.rotorList):
            # mass and moment arm >>> should three-dimensionalize <<<

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
        
        nele = 3
        
        mem0 = fowt.memberList[fowt.nplatmems]
        mem1 = fowt.memberList[fowt.nplatmems+1]

        rTwr1 = mem0.rB - mem0.rA
        rTwr2 = mem1.rB - mem1.rA

        r_node_twr1 = np.zeros([nele+1,3])
        r_node_twr2 = np.zeros([nele+1,3])

        d_node = np.linspace(mem0.d[0], mem0.d[-1], nele+1, endpoint=True)
        t_node = np.linspace(mem0.t[0], mem0.t[-1], nele+1, endpoint=True)

        for i, n in enumerate(np.linspace(0,1,nele+1, endpoint=True)):
            r_node_twr1[i] = mem0.rA + n * rTwr1
            r_node_twr2[i] = mem1.rA + n * rTwr2
        # r_node_twr1 = np.linspace(0,1,nele+1, endpoint=True) * rTwr1 + mem0.rA
        # r_node_twr2 = np.linspace(0,1,nele+1, endpoint=True) * rTwr2 + mem1.rA

        R1 = rotationMatrix(rTwr1)
        R2 = rotationMatrix(rTwr2)

        print('begin nodes;')
        print(f'	structural: 1, static,')
        print(f'            {r_node_twr1[0,0]:.2f}, {r_node_twr1[0,1]:.2f}, {r_node_twr1[0,2]:.2f},')
        print(f'            matr, {R1[0,0]:.3f},{R1[0,1]:.3f},{R1[0,2]:.3f}, {R1[1,0]:.3f},{R1[1,1]:.3f},{R1[1,2]:.3f}, {R1[2,0]:.3f},{R1[2,1]:.3f},{R1[2,2]:.3f},')
        print(f'            null,')
        print(f'            null;\n')

        for i in range(nele):
            print(f'	structural: {i+2}, dynamic,')
            print(f'            {r_node_twr1[i+1,0]:.2f}, {r_node_twr1[i+1,1]:.2f}, {r_node_twr1[i+1,2]:.2f},')
            print(f'            matr, {R1[0,0]:.3f},{R1[0,1]:.3f},{R1[0,2]:.3f}, {R1[1,0]:.3f},{R1[1,1]:.3f},{R1[1,2]:.3f}, {R1[2,0]:.3f},{R1[2,1]:.3f},{R1[2,2]:.3f},')
            print(f'            null,')
            print(f'            null;\n')

        for i in range(nele):
            print(f'	structural: {i+nele+2}, dynamic,')
            print(f'            {r_node_twr2[i+1,0]:.2f}, {r_node_twr2[i+1,1]:.2f}, {r_node_twr2[i+1,2]:.2f},')
            print(f'            matr, {R2[0,0]:.3f},{R2[0,1]:.3f},{R2[0,2]:.3f}, {R2[1,0]:.3f},{R2[1,1]:.3f},{R2[1,2]:.3f}, {R2[2,0]:.3f},{R2[2,1]:.3f},{R2[2,2]:.3f},')
            print(f'            null,')
            print(f'            null;\n')
        print('end: nodes;\n')


        print('begin elements;')
        print(f'	joint: 1, clamp, 1, node, node;\n')

        for i in range(nele):
            A, Iy, Iz, J = section_property(0.5*(d_node[i]+d_node[i+1]), 0.5*(t_node[i]+t_node[i+1]))
            EA = E*A
            EIy = E*Iy
            EIz = E*Iz
            GJ = G*J
            print(f'	beam2: {i+1},')
            print(f'            {i+1}, null,')
            print(f'            {i+2}, null,')
            print(f'            matr, {R1[0,0]:.3f},{R1[0,1]:.3f},{R1[0,2]:.3f}, {R1[1,0]:.3f},{R1[1,1]:.3f},{R1[1,2]:.3f}, {R1[2,0]:.3f},{R1[2,1]:.3f},{R1[2,2]:.3f},')
            print(f'            linear elastic generic,')
            print(f'                diag, {EA:.3e}, {EA:.3e}, {EA:.3e}, {GJ:.3e}, {EIy:.3e}, {EIz:.3e}')
        # print(f"rotationmatrix, {node2}, beam6,")

        # print(f"    {node1}, {node2}, beam6,")
        # print(f"        length, {L:.5f},")
        # print(f"        stiffness, {E*A:.2e}, {E*A:.2e}, {E*A:.2e}, {E*I:.2e}, {E*I:.2e}, {G*J:.2e},")
        # print(f"        damping, 50., 50., 50., 10., 10., 10.,")
        # print(f"        area, {A:.6f},")
        # print(f"        inertia, {I:.6e}, {I:.6e}, {J:.6e},")
        # print(f"        density, {density};\n")

        # tmp = R1 @ np.array([1,0,0])
        # temp = rTwr1 / np.linalg.norm(rTwr1)
        a = 1

        # rot = fowt.rotorList[0]
        # mem = fowt.memberList[fowt.nplatmems]
        # mem.setPosition(r6=fowt.r6)

        # f_aero0[0,:], _, _, _ = rot.calcAero(case, current=False)
        # f_aero0[1,1] = - f_aero0[0,1]
        # f_aero0[1,3] = - f_aero0[0,3]
        # f_aero0[1,5] = - f_aero0[0,5]

        # f_aero_twr_top[0] = transformForce(f_aero0[0,:], offset=(rot.r_hub_rel-mem.rB))
        # f_aero_twr_top[0] = transformForce(f_aero0[0,:], offset=(rot.r_hub_rel-mem.rB))
        # beam = FEModel3D()
        # beam.add_material('Steel', E, G, nu, rho)

        # for ir, rot in enumerate(fowt.rotorList):
        #     # mass and moment arm >>> should three-dimensionalize <<<

        #     mem = fowt.memberList[fowt.nplatmems + ir]
        #     mem.setPosition(r6=fowt.r6)

        #     f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

        #     # counter-clockwise rotating rotor
        #     # An approximate method (needs update)
        #     if ir == 1:
        #           f_aero0[ir,1] = - f_aero0[ir,1]
        #           f_aero0[ir,3] = - f_aero0[ir,3]
        #           f_aero0[ir,5] = - f_aero0[ir,5]

        #     f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
        #     f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG_rel-mem.rB))
            
        #     for il in range(len(mem.ls)):
                
        #         nNode = len(mem.ls)
        #         nn = ir*nNode # ir = 0, nn = 0; ir = 1, nn = nNode
                
        #         if ir == 0:
        #             xs, ys, zs = mem.rA + mem.q*mem.ls[il]
        #             beam.add_node(f'N{il}', xs, ys, zs)
        #         elif ir > 0 and il == 0:
        #             pass
        #         else:
        #             xs, ys, zs = mem.rA + mem.q*mem.ls[il]
        #             beam.add_node(f'N{il+nn-1}', xs, ys, zs)

        #     for il in range(len(mem.ls)-1):
                
        #         nNode = len(mem.ls)
        #         nn = ir*nNode
                
        #         # A0, Iy0, Iz0, J0 = section_property(mem.ds[il], mem.ts[il], type='ellipse')
        #         # A1, Iy1, Iz1, J1 = section_property(mem.ds[il+1], mem.ts[il+1], type='ellipse')
        #         A0, Iy0, Iz0, J0 = section_property(mem.ds[il], mem.ts[il])
        #         A1, Iy1, Iz1, J1 = section_property(mem.ds[il+1], mem.ts[il+1])
        #         A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
                
        #         if ir == 0:
        #             beam.add_section(f'Section{il}', A, Iy, Iz, J, d=mem.ds[il], t=mem.ts[il])

        #         if ir == 1 and il == 0:
        #             beam.add_member(f'M{il+nn-1}', f'N{0}', f'N{il+nn}', 'Steel', f'Section{0}')
                    
        #             # sectional aero loads
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])
                    
        #             #try
        #             # beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])
        #         elif ir == 0:
        #             beam.add_member(f'M{il}', f'N{il}', f'N{il+1}', 'Steel', f'Section{il}')
                    
        #             # sectional aero loads
        #             beam.add_member_dist_load(f'M{il}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
        #             beam.add_member_dist_load(f'M{il}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
        #             beam.add_member_dist_load(f'M{il}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])

        #             #try
        #             # beam.add_member_dist_load(f'M{il}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])
        #         elif ir == 1 and il > 0:
        #             beam.add_member(f'M{il+nn-1}', f'N{il+nn-1}', f'N{il+nn}', 'Steel', f'Section{il}')
                
        #             # sectional aero loads
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FX', mem.F_aero[il][0], mem.F_aero[il+1][0])
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FY', mem.F_aero[il][1], mem.F_aero[il+1][1])
        #             beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][2], mem.F_aero[il+1][2])

        #             #try
        #             # beam.add_member_dist_load(f'M{il+nn-1}', 'FZ', mem.F_aero[il][0], mem.F_aero[il+1][0])

        # d_ten = 0.25
        # A_ten = 0.25*np.pi*d_ten**2
        # beam.add_material('Tension', 2.1e11, 1e-12, 0.3, 1e-6)
        # beam.add_section(f'Section{il+nn+1}', A_ten, 0.0, 0.0, 0.0)

        # beam.add_member(f'M{il+nn}', f'N{int((nn-1)/2)}', f'N{int((3*nn-3)/2)}', 'Tension', f'Section{il+nn+1}', tension_only=True)
        # beam.add_member(f'M{il+nn+1}', f'N{int(nn-1)}',   f'N{int(2*nn-2)}'  , 'Tension', f'Section{il+nn+1}', tension_only=True)
            
        # beam.def_support('N0', True, True, True, True, True, True)
        # beam.def_support(f'N{nn-1}', False, False, False, False, False, False)
        # beam.def_support(f'N{2*nn-2}', False, False, False, False, False, False)

        # beam.def_releases(f'M{il+nn}',   0,0,0,1,1,1,0,0,0,1,1,1)
        # beam.def_releases(f'M{il+nn+1}', 0,0,0,1,1,1,0,0,0,1,1,1)

        
        # beam.add_member_self_weight('FZ', factor=-1)

        # beam.add_node_load(f'N{nn-1}', 'FX', f_aero_twr_top[0][0])
        # beam.add_node_load(f'N{nn-1}', 'FY', f_aero_twr_top[0][1])
        # beam.add_node_load(f'N{nn-1}', 'FZ', f_aero_twr_top[0][2])
        # beam.add_node_load(f'N{nn-1}', 'MX', f_aero_twr_top[0][3])
        # beam.add_node_load(f'N{nn-1}', 'MY', f_aero_twr_top[0][4])
        # beam.add_node_load(f'N{nn-1}', 'MZ', f_aero_twr_top[0][5])

        # beam.add_node_load(f'N{2*nn-2}', 'FX', f_aero_twr_top[1][0])
        # beam.add_node_load(f'N{2*nn-2}', 'FY', f_aero_twr_top[1][1])
        # beam.add_node_load(f'N{2*nn-2}', 'FZ', f_aero_twr_top[1][2])
        # beam.add_node_load(f'N{2*nn-2}', 'MX', f_aero_twr_top[1][3])
        # beam.add_node_load(f'N{2*nn-2}', 'MY', f_aero_twr_top[1][4])
        # beam.add_node_load(f'N{2*nn-2}', 'MZ', f_aero_twr_top[1][5])

        # # beam.analyze()
        # # M_RNA = np.array([rot.mRNA, rot.mRNA, rot.mRNA,  rot.IxRNA, rot.IrRNA, rot.IrRNA])
        # # M_RNA = translateMatrix6to6DOF(M_RNA, [-0.27,0,0])
        
        # # id1 = np.arange((nn-1)*6, nn*6)
        # # id2 = np.arange((2*nn-3)*6, (2*nn-2)*6)
        # # Twr_top_indices = np.hstack((id1, id2))
        # # added_mass_RNA = np.hstack((M_RNA, M_RNA))
        
        # id = [(nn-2)*6, (2*nn-3)*6]    
        # M_RNA = np.array([[350000. , 0.        , 0.       , 0.          , 688446.5   , 0.         ],
        #                   [0.      , 350000.   , 0.       , -688446.5   , 0.         , -144821.25 ],
        #                   [0.      , 0.        , 350000.  , 0.          , 144821.25  , 0.         ],
        #                   [0.      , -688446.5 , 0.       , 45054167.38 , 0.         , 1453861.95 ],
        #                   [688446.5, 0.        , 144821.25, 0.          , 21414090.79, 0.         ],
        #                   [0.      , -144821.25, 0.       , 1453861.95  , 0.         , 20059923.41]])
        # # M_RNA = np.zeros([6,6])
        # fq6, eigen_v = beam.analyze_eigen(id=id, added_mass=M_RNA) # Hz # first 6st mode frequency 
        # eigen_v = np.vstack([np.zeros([6,6]), eigen_v])    
        
        # # static 
        # beam.analyze()

        # # cable stress
        # cable_ele = beam.members[f'M{il+nn+1}']
        # F_cable = cable_ele.axial(0)
        # cable_stress = np.abs(F_cable/A_ten)

        # misses_y=[]
        # misses_z=[]

        # for i, member in enumerate(beam.members.values()):
        #         # if i < 2*nn-2:
        #         if i < nn-1:
        #             x0, y0 = anaylyze_stress(member)
        #             misses_y.append(x0)
        #             misses_z.append(y0)

        #         elif i >= nn-1 and i < 2*nn-2:
                
        #             x0, y0 = anaylyze_stress(member, factor=-1)
        #             misses_y.append(x0)
        #             misses_z.append(y0)

        #         else:
        #             break

        # def plot_mod():
            
        #     nf = eigen_v.shape[1]
        #     DX_mod = np.zeros((len(mem.ls)-1, nf))
        #     DY_mod = np.zeros((len(mem.ls)-1, nf))
        #     DZ_mod = np.zeros((len(mem.ls)-1, nf))
        #     RX_mod = np.zeros((len(mem.ls)-1, nf))
        #     RY_mod = np.zeros((len(mem.ls)-1, nf))
        #     RZ_mod = np.zeros((len(mem.ls)-1, nf))

        #     DX_mod_1 = np.zeros((len(mem.ls)-1, nf))
        #     DY_mod_1 = np.zeros((len(mem.ls)-1, nf))
        #     DZ_mod_1 = np.zeros((len(mem.ls)-1, nf))
        #     RX_mod_1 = np.zeros((len(mem.ls)-1, nf))
        #     RY_mod_1 = np.zeros((len(mem.ls)-1, nf))
        #     RZ_mod_1 = np.zeros((len(mem.ls)-1, nf))
            
        #     for i in range(0, len(mem.ls)-1):
        #          for fs in range(nf):  

        #             # d = beam.members[f'M{i}'].T() @ eigen_v[6*i:6*i+12, fs]
        #             d = eigen_v[6*i:6*i+12, fs]
                    
        #             DX_mod[i][fs] = d[0]
        #             DY_mod[i][fs] = d[1]
        #             DZ_mod[i][fs] = d[2]
        #             RX_mod[i][fs] = d[3]
        #             RY_mod[i][fs] = d[4]
        #             RZ_mod[i][fs] = d[5]    

        #     for i in range(0, len(mem.ls)-1):
        #          for fs in range(nf):  
                    
        #             # d = beam.members[f'M{i+nn-1}'].T() @ eigen_v[6*(i+nn-2):6*(i+nn-2)+12, fs]
        #             if i == 0:
        #                 d = np.zeros(6)
        #             else:
        #                 # d = -beam.members[f'M{i+nn-1}'].T().T @ eigen_v[6*(i+nn-1):6*(i+nn-1)+12, fs]
        #                 d = eigen_v[6*(i+nn-1):6*(i+nn-1)+12, fs]
                    
        #             DX_mod_1[i][fs] = d[0]
        #             DY_mod_1[i][fs] = d[1]
        #             DZ_mod_1[i][fs] = d[2]
        #             RX_mod_1[i][fs] = d[3]
        #             RY_mod_1[i][fs] = d[4]
        #             RZ_mod_1[i][fs] = d[5]   
            
        #     mod_name = ['1st', '2nd', '3rd', '4th','5th','6th']
        #     fig, ax = plt.subplots(6,2)
        #     for iax in range(6):
        #          ax[iax,0].sharey(ax[iax,1]) 
            
        #     ax[0,0].plot(mem.ls[1:], DX_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[0,0].grid()
        #     ax[0,0].set_ylabel('Dx')
        #     ax[0,0].legend()
            
        #     ax[1,0].plot(mem.ls[1:], DY_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[1,0].grid()
        #     ax[1,0].set_ylabel('Dy')
        #     ax[1,0].legend()
            
        #     ax[2,0].plot(mem.ls[1:], DZ_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[2,0].grid()
        #     ax[2,0].set_ylabel('Dz')
        #     ax[2,0].legend()
            
        #     ax[3,0].plot(mem.ls[1:], RX_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[3,0].grid()
        #     ax[3,0].set_ylabel('Rx')
        #     ax[3,0].legend()

        #     ax[4,0].plot(mem.ls[1:], RY_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[4,0].grid()
        #     ax[4,0].set_ylabel('Ry')
        #     ax[4,0].legend()

        #     ax[5,0].plot(mem.ls[1:], RZ_mod[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[5,0].grid()
        #     ax[5,0].set_ylabel('Rz')
        #     ax[5,0].legend()

        #     ax[0,1].plot(mem.ls[1:], DX_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[0,1].grid()
        #     ax[0,1].set_ylabel('Dx')
        #     ax[0,1].legend()
            
        #     ax[1,1].plot(mem.ls[1:], DY_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[1,1].grid()
        #     ax[1,1].set_ylabel('Dy')
        #     ax[1,1].legend()
            
        #     ax[2,1].plot(mem.ls[1:], DZ_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[2,1].grid()
        #     ax[2,1].set_ylabel('Dz')
        #     ax[2,1].legend()
            
        #     ax[3,1].plot(mem.ls[1:], RX_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[3,1].grid()
        #     ax[3,1].set_ylabel('Rx')
        #     ax[3,1].legend()

        #     ax[4,1].plot(mem.ls[1:], RY_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[4,1].grid()
        #     ax[4,1].set_ylabel('Ry')
        #     ax[4,1].legend()

        #     ax[5,1].plot(mem.ls[1:], RZ_mod_1[:,:], label=mod_name, ls='--', linewidth=2)
        #     ax[5,1].grid()
        #     ax[5,1].set_ylabel('Rz')
        #     ax[5,1].legend()
            
        #     return ax 

        # def plot_static():
            
            
        #     My = []
        #     Mz = []
        #     Dx = []
        #     Dy = [] 
        #     Dz = [] 

        #     for i, member in enumerate(beam.members.values()):
        #         # if i < 2*nn-2:
        #         if i < nn-1:

        #             My.append(member.moment('My',0))
        #             Mz.append(member.moment('Mz',0))

        #             Dx.append(member.deflection('dx', 0))
        #             Dy.append(member.deflection('dy', 0))
        #             Dz.append(member.deflection('dz', 0))

        #         elif i >= nn-1 and i < 2*nn-2:
                
        #             My.append(member.moment('My',0))
        #             Mz.append(member.moment('Mz',0))

        #             Dx.append(member.deflection('dx', 0))
        #             Dy.append(member.deflection('dy', 0))
        #             Dz.append(member.deflection('dz', 0))

        #         else:
        #             break
            

        #     # plt.plot(mem.ls[:-1], x[:nn-1])
        #     # plt.plot(mem.ls[:-1], y[:nn-1])
        #     fig, ax = plt.subplots(4,2)
        #     ax[0,0].plot(mem.ls[:-1], misses_y[:nn-1], label='Mises y')
        #     ax[0,0].plot(mem.ls[:-1], misses_z[:nn-1], label='Mises z')
        #     ax[0,0].grid()
        #     ax[0,0].legend()

        #     ax[1,0].plot(mem.ls[:-1], My[:nn-1], label='My')
        #     ax[1,0].plot(mem.ls[:-1], Mz[:nn-1], label='Mz')
        #     ax[1,0].grid()
        #     ax[1,0].legend()

        #     ax[2,0].plot(mem.ls[:-1], Dy[:nn-1], label='Dy')
        #     ax[2,0].grid()
        #     ax[2,0].legend()

        #     ax[3,0].plot(mem.ls[:-1], Dz[:nn-1], label='Dz')
        #     ax[3,0].grid()
        #     ax[3,0].legend()

        #     ax[0,1].plot(mem.ls[:-1], misses_y[nn-1:], label='Mises y')
        #     ax[0,1].plot(mem.ls[:-1], misses_z[nn-1:], label='Mises z')
        #     ax[0,1].grid()
        #     ax[0,1].legend()

        #     ax[1,1].plot(mem.ls[:-1], My[nn-1:], label='My')
        #     ax[1,1].plot(mem.ls[:-1], Mz[nn-1:], label='Mz')
        #     ax[1,1].grid()
        #     ax[1,1].legend()

        #     ax[2,1].plot(mem.ls[:-1], Dy[nn-1:], label='Dy')
        #     ax[2,1].grid()
        #     ax[2,1].legend()

        #     ax[3,1].plot(mem.ls[:-1], Dz[nn-1:], label='Dz')
        #     ax[3,1].grid()
        #     ax[3,1].legend()
            
        #     return ax

        # if plot is True:
        #     ax0 = plot_mod()
        #     ax1 = plot_static()
        #     plt.show()

        # return np.max(misses_y), np.max(misses_z), cable_stress, fq6


def solveTwrOpensees(fowt:FOWT, case, plot=True):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     
        import openseespy.opensees as ops
        # Mbase_list = np.zeros([2, fowt.nrotors])
        # stress_list = np.zeros([2, fowt.nrotors])

        fowt.setPosition([0,0,0,0,0,0])
        fowt.calcTowerAeroLoads(case)

        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)


        for ir, rot in enumerate(fowt.rotorList):
            # mass and moment arm >>> should three-dimensionalize <<<

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG_rel-mem.rB))

        nelem = 20
        nelem_cable = 1

        mem0 = fowt.memberList[fowt.nplatmems]
        # mem1 = fowt.memberList[fowt.nplatmems+1]

        rTwr1 = mem0.rB - mem0.rA
        # rTwr2 = mem1.rB - mem1.rA

        r_node_twr1 = np.zeros([nelem+1,3])
        # r_node_twr2 = np.zeros([nelem+1,3])
        # d_node = np.interp(np.linspace(0, (mem0.stations[-1]-mem0.stations[0]), nelem+1, endpoint=True), mem0.stations, mem0.d)
        # t_node = np.interp(np.linspace(0, (mem0.stations[-1]-mem0.stations[0]), nelem+1, endpoint=True), mem0.stations, mem0.t)
        d_node = np.linspace(mem0.d[0], mem0.d[-1], nelem+1, endpoint=True)
        t_node = np.linspace(mem0.t[0], mem0.t[-1], nelem+1, endpoint=True)

        # ops add node
        for i, n in enumerate(np.linspace(0,1,nelem+1, endpoint=True)):
            r_node_twr1[i] = mem0.rA + n * rTwr1
            # r_node_twr2[i] = mem1.rA + n * rTwr2
            # ops.node(i+1, r_node_twr1[i][0], r_node_twr1[i][1], r_node_twr1[i][2])
        
        r_node = r_node_twr1

        for inode, r in enumerate(r_node):    
            
            ops.node(inode+1, r[0], r[1], r[2])

        # ops fix bottom
        ops.fix(1, 1,1,1, 1,1,1)
            
        # ops add transform

        axis1 = [-1,0,0]
        
        # ops.geomTransf('Linear', 1, *axis1)
        ops.geomTransf('Corotational', 1, *axis1)
        

        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)

        sec_prop = np.zeros([nelem, 6])
        for i in range(nelem):
            A0, Iy0, Iz0, J0 = section_property(d_node[i], t_node[i])
            A1, Iy1, Iz1, J1 = section_property(d_node[i+1], t_node[i+1])

            A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
            sec_prop[i] = [A0, Iy0, Iz0, J0, d_node[i], t_node[i]]

            ops.section('Elastic', i+1, E,A, Iz, Iy, G, J)
            ops.beamIntegration('Legendre', i+1, i+1, 2)

            g = -9.81
            Cd = 1.0
            hHub = fowt.rotorList[0].hHub
            v = case['wind_speed'] * (r_node[i, 2]/hHub)**fowt.shearExp_air

            dis_load_twr = np.array([g*rho*A, 0, 0]) + np.array([0,0,-0.5*1.225*v**2*d_node[i]*Cd])
            
            ops.element('dispBeamColumn', i+1, i+1, i+2, 1, i+1, '-cMass', '-mass', rho*A)

            # ops.element('elasticBeamColumn', i+1, i+1, i+2, A, E, G, J, Iy, Iz, 1, '-mass', rho*A, '-cMass')
            ops.eleLoad('-ele', i+1, '-type','-beamUniform', dis_load_twr[1], dis_load_twr[2], dis_load_twr[0])

        mass_trans = 3.5e5
        inertia1 = 4.37e7
        inertia2 = 2.353e7
        inertia3 = 2.542e7

        # mass_trans = 945914.1459
        # inertia1 = 378338268.3
        # inertia2 = 271546203.7
        # inertia3 = 252479342.8

        ops.mass(nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)

        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.test('NormDispIncr', 1e-6, 10)
        ops.algorithm('ModifiedNewton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')

        # rotor load
        ops.load(nelem+1, *f_aero_twr_top[0])

        import opsvis as opsv    

        ops.analyze(1)

        tmp = ops.nodeDisp(nelem+1)[:3]
        print(f'node Dispx of top = {tmp[0]:.4e}')
        print(f'node Dispy of top = {tmp[1]:.4e}')


        def calc_stress(ele=0, secType='circular', n=200, plot_stress=True):
            
            Fx, Fy, Fz, Mx, My, Mz = ops.eleResponse(ele+1, 'localForce')[:6]

            print(f'base Fx = {Fz:.4e}')
            print(f'base Fy = {Fy:.4e}')
            
            print(f'base Mx = {Mz:.4e}')
            print(f'base My = {My:.4e}')

            A, Iy, Iz, J, d, t = sec_prop[ele,:]

            # if secType == 'circular':
            #     Sz = Sy = 2 * ((d-t)/2)**2 * t
            #     y = z = 0.5*d
            # elif secType == 'ellipse':
            #     NotImplementedError()
            r = 0.5*d

            def compute_vom_mises(theta, r):
                """
                Compute the second deviatoric stress invariant J2 from a stress tensor 
                given in Voigt notation (assuming shear components are not multiplied by 2).

                Parameters:
                    sigma_voigt: array-like of length 6
                        The stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

                Returns:
                    J2: float
                        The second invariant of the deviatoric stress tensor
                """
                y = r*np.cos(theta)
                z = r*np.sin(theta)

                sigma_xx = Fx/A - My*z/Iy + Mz*y/Iz
                sigma_yy = 0
                sigma_zz = 0  

                # sigma_xy = Fy/(0.25*np.pi*d**2)/2/t
                # sigma_xz = Fz/(0.25*np.pi*d**2)/2/t
                Qz = r**2*t*np.cos(theta)
                Qy = r**2*t*np.sin(theta)
                sigma_xy = Fy*Qz/Iz/t
                sigma_xz = Fz*Qy/Iy/t
                sigma_yz = Mx*r/J
                
                s = np.zeros(6)
                mean_stress = np.mean([sigma_xx, sigma_yy, sigma_zz])
                s[:3] = np.array([sigma_xx, sigma_yy, sigma_zz]) - mean_stress
                s[3:] = np.array([sigma_xy, sigma_xz, sigma_yz])

                J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2 +
                            2 * (s[3]**2 + s[4]**2 + s[5]**2))
                
                von_mises = np.sqrt(3*J2)
                # von_mises = Qy
                
                return von_mises

            # alpha = np.arctan2(-My/Iz, Mz/Iy)
            # alpha = np.arctan2(Mz/Iy, My/Iz)

            from scipy.optimize import minimize_scalar
            def objective_function(theta_rad, *args):

                return -compute_vom_mises(theta_rad, *args)
            
            result = minimize_scalar(objective_function, bounds=(0, 2 * np.pi), method='bounded', args=r)
            
            alpha = result.x
            max_von_mises = compute_vom_mises(result.x, r)
            # max_von_mises1 = compute_vom_mises(alpha, r)

            if plot_stress:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import cm
                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection

                theta = np.linspace(0, np.pi*2, n, endpoint=True)

                von_mises = [compute_vom_mises(theta[i], r) for i in range(n)]

                factor = 10
                pts_out = np.array([r*np.cos(theta), r*np.sin(theta)])
                pts_in = np.array([(r-t*factor)*np.cos(theta), (r-t*factor)*np.sin(theta)])

                patches = []
                colors = []

                for i in range(n-1):
                    quad = [pts_in[:,   i].tolist(), 
                            pts_in[:, i+1].tolist(), 
                            pts_out[:,i+1].tolist(), 
                            pts_out[:,  i].tolist()]

                    polygon = Polygon(quad, closed=True)
                    patches.append(polygon)
                    colors.append(von_mises[i])

                fig, ax = plt.subplots(figsize=(6,6))
                p = PatchCollection(patches, cmap='coolwarm', edgecolor='k', alpha=0.9, linewidth=0.0)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax, label='Shear Stress')

                ax.set_aspect('equal')
                ax.set_xlim(-1.2*r, 1.2*r)
                ax.set_ylim(-1.2*r, 1.2*r)
                ax.set_title('Thin-Walled Circular Section with Shear Stress Distribution')

            
            return max_von_mises
        
        # von_mises = np.zeros(nelem)
        # for i in range(nelem):
        #     von_mises[i] = calc_stress(i, plot_stress=False)

        # max von_mises in twr bottom
        von_mises_bot = calc_stress(0, plot_stress=True)
        plt.show()
        # eigen anlysis
        num_modes = 6
        eigenvals = ops.eigen(num_modes)
        freqs = [np.sqrt(lam)/(2*np.pi) for lam in eigenvals]

        print("\nFrequency(Hz):")
        for i, f in enumerate(freqs):
            print(f"Mode {i+1}: {f:.4f} Hz")

        
        # mod = np.zeros([num_modes, 2*nelem+1, 6])
        # disp = np.zeros([2*nelem+1, 6])
        # cable_tension = np.zeros(2)

        # for i in range(2*nelem+1):
        #     for j in range(num_modes):
        #         mod[j,i,:] = ops.nodeEigenvector(i+1, j+1)
        #         # mod[j,i,3:] = mod[j,i,3:] * 180/np.pi
            
        #     disp[i] = ops.nodeDisp(i+1)
        #     # disp[i,3:] = disp[i,3:] * 180/np.pi
        
        # # tmp = ops.eleForce(1000)
        # cable_tension[0] = ops.eleForce(1000, 2)
        # cable_tension[1] = ops.eleForce(1001, 2)

        # num_modes = 6
        # eigenvals = ops.eigen(num_modes)
        # freqs = [np.sqrt(lam)/(2*np.pi) for lam in eigenvals]

        # print("\nFrequency(Hz):")
        # for i, f in enumerate(freqs):
        #     print(f"Mode {i+1}: {f:.4f} Hz")

        import opsvis as opsv
        fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
            'marker': '', 'markersize': 6}
        opsv.plot_mode_shape(2, interpFlag=1, fmt_defo=fmt_defo)
        plt.show()
        a = 1   


def solveTwrCombinationOpensees(fowt:FOWT, case, plot=True, nelem=20):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     
        import openseespy.opensees as ops

        fowt.setPosition([0,0,0,0,0,0])
        fowt.calcTowerAeroLoads(case)

        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)


        for ir, rot in enumerate(fowt.rotorList):

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG_rel-mem.rB))

        mem0 = fowt.memberList[fowt.nplatmems]
        mem1 = fowt.memberList[fowt.nplatmems+1]

        rTwr1 = mem0.rB - mem0.rA
        rTwr2 = mem1.rB - mem1.rA

        r_node_twr1 = np.zeros([nelem+1,3])
        r_node_twr2 = np.zeros([nelem+1,3])

        d_node = np.linspace(mem0.d[0], mem0.d[-1], nelem+1, endpoint=True)
        t_node = np.linspace(mem0.t[0], mem0.t[-1], nelem+1, endpoint=True)

        # ops add nodes
        for i, n in enumerate(np.linspace(0,1,nelem+1, endpoint=True)):
            r_node_twr1[i] = mem0.rA + n * rTwr1
            r_node_twr2[i] = mem1.rA + n * rTwr2
        
        r_node = np.concatenate([r_node_twr1, r_node_twr2[1:]], axis=0)

        for inode, r in enumerate(r_node):    
            
            ops.node(inode+1, r[0], r[1], r[2])

        # ops fix bottom
        ops.fix(1, 1,1,1, 1,1,1)
            
        # ops add transform
        axis1 = np.cross([1,0,0], rTwr1)
        axis1 = axis1 / np.linalg.norm(axis1)
        R1 = rotationMatrix(rTwr1, ref=[-1,0,0])
        
        axis2 = np.cross([-1,0,0], rTwr2)
        axis2 = axis2 / np.linalg.norm(axis2)
        R2 = rotationMatrix(rTwr2, ref=[1,0,0])
        
        axis3 = [0, 0, 1]

        # ops.geomTransf('PDelta', 1, *axis1)
        # ops.geomTransf('PDelta', 2, *axis2)  
        # ops.geomTransf('PDelta', 3, *axis3) 

        ops.geomTransf('Corotational', 1, *axis1)
        ops.geomTransf('Corotational', 2, *axis2)  
        ops.geomTransf('Corotational', 3, *axis3)
        

        # load patern
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)

        sec_prop = np.zeros([nelem, 6])
        # ops add elements
        for i in range(nelem):
            A0, Iy0, Iz0, J0 = section_property(d_node[i], t_node[i])
            A1, Iy1, Iz1, J1 = section_property(d_node[i+1], t_node[i+1])

            A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
            sec_prop[i] = [A0, Iy0, Iz0, J0, d_node[i], t_node[i]]

            ops.section('Elastic', i+1, E,A, Iz, Iy, G, J)
            ops.beamIntegration('Lobatto', i+1, i+1, 5)

            g = -9.81
            Cd = 1.0
            hHub = fowt.rotorList[0].hHub
            v = case['wind_speed'] * (r_node[i, 2]/hHub)**fowt.shearExp_air

            dis_load_twr1 = R1.T @ (np.array([0, 0, g*rho*A]) + np.array([0.5*1.225*v**2*d_node[i]*Cd,0,0]))
            dis_load_twr2 = R2.T @ (np.array([0, 0, g*rho*A]) + np.array([0.5*1.225*v**2*d_node[i]*Cd,0,0]))
            
            ops.element('dispBeamColumn', i+1, i+1, i+2, 1, i+1, '-cMass', '-mass', rho*A)

            # ops.element('elasticBeamColumn', i+1, i+1, i+2, A, E, G, J, Iy, Iz, 1, '-mass', rho*A, '-cMass')
            ops.eleLoad('-ele', i+1, '-type','-beamUniform', dis_load_twr1[1], dis_load_twr1[2], dis_load_twr1[0])

            if i == 0:
                ops.element('dispBeamColumn', i+nelem+1, i+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
            else:
                ops.element('dispBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
                # print(f'node1:{i+nelem+1},  node2:{i+nelem+2}')
        
        # add truss element representing cable
        ops.uniaxialMaterial('Elastic', 200, E)
        ops.element('corotTruss', 1000, nelem+1, 2*nelem+1, 0.25, 200, '-rho', rho*1e-12)
        ops.element('corotTruss', 1001, (nelem+3)//2, (nelem+3)//2+nelem, 0.25, 200, '-rho', rho*1e-12)
        
        # addition mass and inertia from naccele and rotor
        mass_trans = 3.5e5
        inertia1 = 4.37e7
        inertia2 = 2.353e7
        inertia3 = 2.542e7

        ops.mass(nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        ops.mass(2*nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.test('NormDispIncr', 1e-6, 10)
        ops.algorithm('ModifiedNewton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')

        # rotor load
        ops.load(nelem+1, *f_aero_twr_top[0])
        ops.load(2*nelem+1, *f_aero_twr_top[1])

        import opsvis as opsv    

        ops.analyze(1)
        twr_top_disp_x = ops.nodeDisp(nelem+1)[0]
        print(f'node Dispx of top = {twr_top_disp_x:.3e}')

        def calc_stress(ele=0, secType='circular', n=200, plot_stress=True):
            
            Fx, Fy, Fz, Mx, My, Mz = ops.eleResponse(ele+1, 'localForce')[:6]

            A, Iy, Iz, J, d, t = sec_prop[ele,:]

            # if secType == 'circular':
            #     Sz = Sy = 2 * ((d-t)/2)**2 * t
            #     y = z = 0.5*d
            # elif secType == 'ellipse':
            #     NotImplementedError()
            r = 0.5*d

            def compute_vom_mises(r, theta):
                """
                Compute the second deviatoric stress invariant J2 from a stress tensor 
                given in Voigt notation (assuming shear components are not multiplied by 2).

                Parameters:
                    sigma_voigt: array-like of length 6
                        The stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

                Returns:
                    J2: float
                        The second invariant of the deviatoric stress tensor
                """
                y = r*np.cos(theta)
                z = r*np.sin(theta)

                sigma_xx = Fx/A - My*z/Iy + Mz*y/Iz
                sigma_yy = 0
                sigma_zz = 0  

                sigma_xy = Fy/(0.25*np.pi*d**2)/2/t
                sigma_xz = Fz/(0.25*np.pi*d**2)/2/t
                sigma_yz = Mx*r/J
                
                s = np.zeros(6)
                mean_stress = np.mean([sigma_xx, sigma_yy, sigma_zz])
                s[:3] = np.array([sigma_xx, sigma_yy, sigma_zz]) - mean_stress
                s[3:] = np.array([sigma_xy, sigma_xz, sigma_yz])

                J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2 +
                            2 * (s[3]**2 + s[4]**2 + s[5]**2))
                
                von_mises = np.sqrt(3*J2)
                
                return von_mises

            alpha = np.arctan2(-My/Iz, Mz/Iy)
            # alpha = np.arctan2(Mz/Iy, My/Iz)

            max_von_mises = compute_vom_mises(r, alpha)

            if plot_stress:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import cm
                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection

                theta = np.linspace(0, np.pi*2, n, endpoint=True)

                von_mises = [compute_vom_mises(r, theta[i]) for i in range(n)]

                factor = 10
                pts_out = np.array([r*np.cos(theta), r*np.sin(theta)])
                pts_in = np.array([(r-t*factor)*np.cos(theta), (r-t*factor)*np.sin(theta)])

                patches = []
                colors = []

                for i in range(n-1):
                    quad = [pts_in[:,   i].tolist(), 
                            pts_in[:, i+1].tolist(), 
                            pts_out[:,i+1].tolist(), 
                            pts_out[:,  i].tolist()]

                    polygon = Polygon(quad, closed=True)
                    patches.append(polygon)
                    colors.append(von_mises[i])

                fig, ax = plt.subplots(figsize=(6,6))
                p = PatchCollection(patches, cmap='coolwarm', edgecolor='k', alpha=0.9, linewidth=0.0)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax, label='Shear Stress')

                ax.set_aspect('equal')
                ax.set_xlim(-1.2*r, 1.2*r)
                ax.set_ylim(-1.2*r, 1.2*r)
                ax.set_title('Thin-Walled Circular Section with Shear Stress Distribution')

            
            return max_von_mises
        
        # von_mises = np.zeros(nelem)
        # for i in range(nelem):
        #     von_mises[i] = calc_stress(i, plot_stress=False)

        # max von_mises in twr bottom
        von_mises_bot = calc_stress(0, plot_stress=False)
        # plt.show()
        # eigen anlysis
        num_modes = 6
        eigenvals = ops.eigen(num_modes)
        freqs = [np.sqrt(lam)/(2*np.pi) for lam in eigenvals]

        print("\nFrequency(Hz):")
        for i, f in enumerate(freqs):
            print(f"Mode {i+1}: {f:.4f} Hz")

        
        mod = np.zeros([num_modes, 2*nelem+1, 6])
        disp = np.zeros([2*nelem+1, 6])
        cable_tension = np.zeros(2)

        for i in range(2*nelem+1):
            for j in range(num_modes):
                mod[j,i,:] = ops.nodeEigenvector(i+1, j+1)
                # mod[j,i,3:] = mod[j,i,3:] * 180/np.pi
            
            disp[i] = ops.nodeDisp(i+1)
            # disp[i,3:] = disp[i,3:] * 180/np.pi
        
        # tmp = ops.eleForce(1000)
        cable_tension[0] = ops.eleForce(1000, 2)
        cable_tension[1] = ops.eleForce(1001, 2)
        

        def plot_static_disp(factor=1e4):

            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection


            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            l = np.linalg.norm(rTwr1) * np.linspace(0,1, nelem+1, endpoint=True)

            colors = cm.inferno(np.linspace(0.15, 0.7, 6))

            lines = [disp[:nelem+1, i]*factor for i in range(6)]

            for i in range(6):
                ax.plot(np.ones_like(l)*(i+1) , l , lines[i], color=colors[i], label=f'mod{i}', linewidth=2.0, linestyle='--')

                verts = [list(zip(np.ones_like(l)*(i+1), l, np.full_like(l, lines[i])))] 
                verts[0].append([i+1,l[-1],0]) 
                verts[0].append([i+1,0,0])
                poly = Poly3DCollection(verts, facecolors=colors[i], alpha=0.15)  # 创建多边形集合

                ax.add_collection3d(poly, zs=-1, zdir='y')  # 添加到 3D 图中
                ax.invert_xaxis()
                ax.grid()
                ax.legend()
                ax.set_zlabel(f'disp*{factor:.0e}')
                ax.set_xlabel(f'Mods')
                ax.set_ylabel(f'length')

            return ax


        def plot_eigen_disp(nf=0, factor=1e4):

            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection


            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            l = np.linalg.norm(rTwr1) * np.linspace(0,1, nelem+1, endpoint=True)

            colors = cm.inferno(np.linspace(0.15, 0.7, 6))

            lines = [np.abs(mod[nf, :nelem+1, i])*factor for i in range(6)]

            for i in range(6):
                ax.plot(np.ones_like(l)*(i+1) , l , lines[i], color=colors[i], label=f'mod{i}', linewidth=2.0, linestyle='--')

                verts = [list(zip(np.ones_like(l)*(i+1), l, np.full_like(l, lines[i])))] 
                verts[0].append([i+1,l[-1],0]) 
                verts[0].append([i+1,0,0])
                poly = Poly3DCollection(verts, facecolors=colors[i], alpha=0.15)  # 创建多边形集合

                ax.add_collection3d(poly, zs=-1, zdir='y')  # 添加到 3D 图中
                ax.invert_xaxis()
                ax.grid()
                ax.legend()
                ax.set_zlabel(f'disp*{factor:.0e}')
                ax.set_xlabel(f'Mods')
                ax.set_ylabel(f'length')

            return ax


        def plot_static_3D(factor=[1e4,1e3,1e4], plot_rotor=True, plot_tower=True, plot_cable=True):

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            if plot_tower:          
                mem0.plot(ax)
                mem1.plot(ax)
            else:
                ax.plot(r_node_twr1[:,0],r_node_twr1[:,1],r_node_twr1[:,2], linewidth=5.0, color='grey')
                ax.plot(r_node_twr2[:,0],r_node_twr2[:,1],r_node_twr2[:,2], linewidth=5.0, color='grey')

            if plot_cable:
                ax.plot([r_node_twr1[-1,0],r_node_twr2[-1,0]],
                        [r_node_twr1[-1,1],r_node_twr2[-1,1]],
                        [r_node_twr2[-1,2],r_node_twr1[-1,2]], 
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
        
                ax.plot([r_node_twr1[(nelem+1)//2,0],r_node_twr2[(nelem+1)//2,0]],
                        [r_node_twr1[(nelem+1)//2,1],r_node_twr2[(nelem+1)//2,1]],
                        [r_node_twr2[(nelem+1)//2,2],r_node_twr1[(nelem+1)//2,2]],
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
            
            twr1_mod = r_node_twr1[:,:3]+disp[:nelem+1, :3] * factor
            twr2_mod = r_node_twr2[:,:3]+np.vstack([disp[0, :3], disp[nelem+1:, :3]]) * factor

            ax.plot(twr1_mod[:,0],
                    twr1_mod[:,1],
                    twr1_mod[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
        
            ax.plot(twr2_mod[:,0],
                    twr2_mod[:,1],
                    twr2_mod[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
            
            if plot_rotor:
                fowt.rotorList[0].plot(ax, airfoils=False, draw_circle=True)
                fowt.rotorList[1].plot(ax, airfoils=False, draw_circle=True)

            '''
            plot cable after deformation
            # ax.plot([twr1_mod1[-1,0],twr2_mod1[-1,0]],
            #         [twr1_mod1[-1,1],twr2_mod1[-1,1]],
            #         [twr1_mod1[-1,2],twr2_mod1[-1,2]], 
            #         linewidth=2.0, 
            #         color='r', 
            #         linestyle='--', 
            #         alpha=0.8)

            # ax.plot([twr1_mod1[(nelem+1)//2,0],twr2_mod1[(nelem+1)//2,0]],
            #         [twr1_mod1[(nelem+1)//2,1],twr2_mod1[(nelem+1)//2,1]],
            #         [twr1_mod1[(nelem+1)//2,2],twr2_mod1[(nelem+1)//2,2]],
            #           linewidth=2.0, 
            #           color='r', 
            #           linestyle='--', 
            #           alpha=0.8)

            '''

            '''
            # nn = 10
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn] * 0.5


            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')


            # nn = 12
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn]*0.5

            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')
            '''
            
            def set_axes_equal(ax):
                '''Set 3D plot axes to equal scale'''
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                max_range = max(x_range, y_range, z_range)

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
                ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
                ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

            set_axes_equal(ax)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Structure View: Two Towers + Cable')
            ax.view_init(elev=20, azim=45)
            ax.grid(True)

            return ax


        def plot_eigen_3D(nf=0, factor=[1e4,1e3,1e4], plot_rotor=True, plot_tower=True, plot_cable=True):

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            if plot_tower:          
                mem0.plot(ax)
                mem1.plot(ax)
            else:
                ax.plot(r_node_twr1[:,0],r_node_twr1[:,1],r_node_twr1[:,2], linewidth=5.0, color='grey')
                ax.plot(r_node_twr2[:,0],r_node_twr2[:,1],r_node_twr2[:,2], linewidth=5.0, color='grey')

            if plot_cable:
                ax.plot([r_node_twr1[-1,0],r_node_twr2[-1,0]],
                        [r_node_twr1[-1,1],r_node_twr2[-1,1]],
                        [r_node_twr2[-1,2],r_node_twr1[-1,2]], 
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
        
                ax.plot([r_node_twr1[(nelem+1)//2,0],r_node_twr2[(nelem+1)//2,0]],
                        [r_node_twr1[(nelem+1)//2,1],r_node_twr2[(nelem+1)//2,1]],
                        [r_node_twr2[(nelem+1)//2,2],r_node_twr1[(nelem+1)//2,2]],
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
            
            if nf > 1 & nf < 6:
                factor[0] *= 100
            twr1_mod1 = r_node_twr1[:,:3]+mod[nf, :nelem+1, :3] * factor
            twr2_mod1 = r_node_twr2[:,:3]+np.vstack([mod[nf, 0, :3], mod[nf, nelem+1:, :3]]) * factor

            ax.plot(twr1_mod1[:,0],
                    twr1_mod1[:,1],
                    twr1_mod1[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
        
            ax.plot(twr2_mod1[:,0],
                    twr2_mod1[:,1],
                    twr2_mod1[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
            
            if plot_rotor:
                fowt.rotorList[0].plot(ax, airfoils=False, draw_circle=True)
                fowt.rotorList[1].plot(ax, airfoils=False, draw_circle=True)

            '''
            plot cable after deformation
            # ax.plot([twr1_mod1[-1,0],twr2_mod1[-1,0]],
            #         [twr1_mod1[-1,1],twr2_mod1[-1,1]],
            #         [twr1_mod1[-1,2],twr2_mod1[-1,2]], 
            #         linewidth=2.0, 
            #         color='r', 
            #         linestyle='--', 
            #         alpha=0.8)

            # ax.plot([twr1_mod1[(nelem+1)//2,0],twr2_mod1[(nelem+1)//2,0]],
            #         [twr1_mod1[(nelem+1)//2,1],twr2_mod1[(nelem+1)//2,1]],
            #         [twr1_mod1[(nelem+1)//2,2],twr2_mod1[(nelem+1)//2,2]],
            #           linewidth=2.0, 
            #           color='r', 
            #           linestyle='--', 
            #           alpha=0.8)

            '''

            '''
            # nn = 10
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn] * 0.5


            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')


            # nn = 12
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn]*0.5

            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')
            '''
            
            def set_axes_equal(ax):
                '''Set 3D plot axes to equal scale'''
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                max_range = max(x_range, y_range, z_range)

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
                ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
                ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

            set_axes_equal(ax)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Structure View: Two Towers + Cable')
            ax.view_init(elev=20, azim=45)
            ax.grid(True)

        # plt.tight_layout()
        # plt.show()

            return ax

        # ax = plot_eigen_disp(nf=2, factor=1e4)
        # ax = plot_static_disp(factor=1)
        # ax = plot_eigen_3D(nf=5, factor=[1e5,1e4,1e4])
        # ax = plot_static_3D(factor=[2e1,1e3,1e2])
        # plt.tight_layout()
        # plt.show()

        if plot:
            ax0 = plot_eigen_3D(0, factor=[1e4,1e4,1e4])
            ax1 = plot_eigen_3D(1, factor=[1e4,1e4,1e4])
            ax2 = plot_eigen_3D(2, factor=[1e2,1e4,1e4])
            ax3 = plot_eigen_3D(3, factor=[1e4,1e4,1e4])
            ax4 = plot_eigen_3D(4, factor=[1e4,1e4,1e4])
            ax5 = plot_eigen_3D(5, factor=[1e4,1e4,1e4])
            ax6 = plot_static_3D(factor=[2e1,1e3,1e2])

            return von_mises_bot, twr_top_disp_x, freqs, [ax0,ax1,ax2,ax3,ax4,ax5,ax6]
        else:

            return von_mises_bot, twr_top_disp_x, freqs


def section_property(d, t, type='circular'):
            
    if type == 'circular':
    
        A = 0.25*np.pi*(d**2-(d-2*t)**2)
        Iy = Iz = 1/64 * np.pi * (d**4 - (d-2*t)**4)
        J = 1/32 * np.pi * (d**4 - (d-2*t)**4)
    
    elif type == 'ellipse':
        
        a = d
        b = 0.5*d

        A = np.pi*(a*b-(a-t)*(b-t))
        Iy = 0.25*np.pi*(a**3*b - (a-t)**3 * (b-t))
        Iz = 0.25*np.pi*(a*b**3 - (a-t) * (b-t)**3)
        J = Iy+Iz

    else:
        raise NotImplementedError()
    
    return A, Iy, Iz, J

def anaylyze_stress(member:PhysMember, Stype='circular', factor=1):
            
    d = member.section.d
    t = member.section.t
    A = member.section.A
    Iy = member.section.Iy
    Iz = member.section.Iz
    J = member.section.J
    
    if Stype == 'circular':
        Sz = Sy = 2 * ((d-t)/2)**2 * t
        y = z = d
    elif Stype == 'ellipse':
        NotImplementedError()
    
    Fx = member.axial(0)
    sigmax = Fx/A
    
    My = member.moment('My', 0)
    Mz = member.moment('Mz', 0)*factor
    sigmay = My * z / Iy
    sigmaz = Mz * y / Iz
    
    Fy = member.shear('Fy', 0)
    Fz = member.shear('Fz', 0)

    tauy = Fy * Sy / Iy / (2*t)
    tauz = Fz * Sz / Iz / (2*t)
    
    T = member.torque(0)
    tauty = T*y/J
    tautz = T*z/J
    
    # sigma_mises_y = np.sqrt( (sigmay)**2)
    # sigma_mises_z = np.sqrt( (sigmaz)**2)
    sigma_mises_y = np.sqrt( (sigmax + sigmay)**2 + 3*(tauz**2+tauty**2) )
    sigma_mises_z = np.sqrt( (sigmax + sigmaz)**2 + 3*(tauy**2+tautz**2) )
    
    return sigma_mises_y, sigma_mises_z

def adjust_ballast_opt(fowt:FOWT, l_fill=0.01, iter=100):
    
    fowt_ = fowt.copy()
    
    fowt_.setPosition([0,0,0,0,0,0])
    fowt_.calcStatics()
    
    xCG = fowt_.rCG[0]
    xCB = fowt_.rCB[0]

    offset0 = xCB - xCG
    try:
        offset = offset0
        for i in range(iter):
            
            if offset >= 0:
                fowt_.memberList[4].l_fill -= l_fill
                fowt_.memberList[5].l_fill -= l_fill
            else:
                fowt_.memberList[6].l_fill -= l_fill

            fowt_.calcStatics()
            xCG = fowt_.rCG[0]
            xCB = fowt_.rCB[0]
            offset = abs(xCB - xCG)
            
            if abs(offset < 1e-2) or offset * offset0 < 0:
                break
    
    except:

        RuntimeError(f'iteration procedure fails in iter{i}, offset = {offset}[m], please check the initial design')
    
    return fowt_

def adjust_ballast_opt_up(fowt:FOWT, l_fill=0.01, iter=100):
    
    fowt_ = fowt.copy()
    
    fowt_.setPosition([0,0,0,0,0,0])
    fowt_.calcStatics()
    
    xCG = fowt_.rCG[0]
    xCB = fowt_.rCB[0]

    offset0 = xCG - xCB
    try:
        offset = offset0
        for i in range(iter):
            
            if offset >= 0:
                fowt_.memberList[4].l_fill -= l_fill
                fowt_.memberList[5].l_fill -= l_fill
            else:
                fowt_.memberList[6].l_fill -= l_fill

            fowt_.calcStatics()
            xCG = fowt_.rCG[0]
            xCB = fowt_.rCB[0]
            offset = xCG - xCB
            
            if abs(offset) < 1e-2 or offset * offset0 < 0:
                print(f'balanced after {i+1} iterations')
                print(f'{l_fill *(i+1) * fowt_.memberList[4].sl[0,0]* fowt_.memberList[4].sl[0,1] * fowt_.rho_water / 1e3:.2f} t mass has been balanced')
                print(f'{l_fill *(i+1)} m ballast has been balanced')
                break
    
    except:

        RuntimeError(f'iteration procedure fails in iter{i}, offset = {offset}[m], please check the initial design')
    
    return fowt_

# def adjust_Column_ballast_opt(fowt:FOWT, l_fill=0.01, iter=100, case=None):
    
#     fowt_ = fowt.copy()
    
#     fowt_.setPosition([0,0,0,0,0,0])
#     fowt_.calcStatics()
#     fowt_.solveStatics(case)

#     heave_offset_0 = fowt_.Xi0[2]
#     try:
#         offset = heave_offset_0
#         for i in range(iter):
            
#             if offset >= 0:
#                 fowt_.memberList[1].l_fill += l_fill
#                 fowt_.memberList[2].l_fill += l_fill
#                 fowt_.memberList[2].l_fill += l_fill
#             else:
#                 fowt_.memberList[1].l_fill -= l_fill
#                 fowt_.memberList[2].l_fill -= l_fill
#                 fowt_.memberList[2].l_fill -= l_fill

#             fowt_.setPosition([0,0,0,0,0,0])
#             fowt_.solveStatics(case)
#             offset = fowt_.Xi0[2]
            
#             if abs(offset) < 1e-2 or offset * heave_offset_0 < 0:
#                 break
    
#     except:

#         RuntimeError(f'iteration procedure fails in iter{i}, offset = {offset}[m], please check the initial design')
    
#     return fowt_

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert np.array into list
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}  # recursive
    elif isinstance(obj, list):
        return [convert_numpy_to_list(i) for i in obj]  # recursive lists
    else:
        return obj
    
def rotationMatrix(r, ref=None):
    if ref is None:
        ref = np.array([0.0, 0.0, 1.0])
    
    q = r / np.linalg.norm(r)
    
    if np.isclose(abs(np.dot(r, ref)), 1.0, atol=1e-8):
        ref = np.array([0.0, 1.0, 0.0])
    
    y_local = ref - np.dot(ref, q) * q
    y_local = y_local / np.linalg.norm(y_local)
    z_local = np.cross(q, y_local)
    
    R = np.column_stack((q, y_local, z_local))
    return R


def solveTwrCombinationSpectrumOpensees(fowt:FOWT, case, plot=True, nelem=20, direction=0, Tn=None, Sa=None):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     
        import openseespy.opensees as ops

        fowt.setPosition([0,0,0,0,0,0])
        fowt.calcTowerAeroLoads(case)

        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)


        for ir, rot in enumerate(fowt.rotorList):

            mem = fowt.memberList[fowt.nplatmems + ir]
            mem.setPosition(r6=fowt.r6)

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG_rel-mem.rB))

        mem0 = fowt.memberList[fowt.nplatmems]
        mem1 = fowt.memberList[fowt.nplatmems+1]

        rTwr1 = mem0.rB - mem0.rA
        rTwr2 = mem1.rB - mem1.rA

        r_node_twr1 = np.zeros([nelem+1,3])
        r_node_twr2 = np.zeros([nelem+1,3])

        d_node = np.linspace(mem0.d[0], mem0.d[-1], nelem+1, endpoint=True)
        t_node = np.linspace(mem0.t[0], mem0.t[-1], nelem+1, endpoint=True)

        # ops add nodes
        for i, n in enumerate(np.linspace(0,1,nelem+1, endpoint=True)):
            r_node_twr1[i] = mem0.rA + n * rTwr1
            r_node_twr2[i] = mem1.rA + n * rTwr2
        
        r_node = np.concatenate([r_node_twr1, r_node_twr2[1:]], axis=0)

        for inode, r in enumerate(r_node):    
            
            ops.node(inode+1, r[0], r[1], r[2])

        # ops fix bottom
        ops.fix(1, 1,1,1, 1,1,1)
            
        # ops add transform
        axis1 = np.cross([1,0,0], rTwr1)
        axis1 = axis1 / np.linalg.norm(axis1)
        R1 = rotationMatrix(rTwr1, ref=[-1,0,0])
        
        axis2 = np.cross([-1,0,0], rTwr2)
        axis2 = axis2 / np.linalg.norm(axis2)
        R2 = rotationMatrix(rTwr2, ref=[1,0,0])
        
        axis3 = [0, 0, 1]

        # ops.geomTransf('PDelta', 1, *axis1)
        # ops.geomTransf('PDelta', 2, *axis2)  
        # ops.geomTransf('PDelta', 3, *axis3) 

        ops.geomTransf('Corotational', 1, *axis1)
        ops.geomTransf('Corotational', 2, *axis2)  
        ops.geomTransf('Corotational', 3, *axis3) 

        sec_prop = np.zeros([nelem, 6])
        # ops add elements
        for i in range(nelem):
            A0, Iy0, Iz0, J0 = section_property(d_node[i], t_node[i])
            A1, Iy1, Iz1, J1 = section_property(d_node[i+1], t_node[i+1])

            A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
            sec_prop[i] = [A0, Iy0, Iz0, J0, d_node[i], t_node[i]]

            ops.section('Elastic', i+1, E,A, Iz, Iy, G, J)
            ops.beamIntegration('Lobatto', i+1, i+1, 5)

            g = -9.81
            Cd = 1.0
            hHub = fowt.rotorList[0].hHub
            v = case['wind_speed'] * (r_node[i, 2]/hHub)**fowt.shearExp_air

            dis_load_twr1 = R1.T @ (np.array([0, 0, g*rho*A]) + np.array([0.5*1.225*v**2*d_node[i]*Cd,0,0]))
            dis_load_twr2 = R2.T @ (np.array([0, 0, g*rho*A]) + np.array([0.5*1.225*v**2*d_node[i]*Cd,0,0]))
            
            ops.element('dispBeamColumn', i+1, i+1, i+2, 1, i+1, '-cMass', '-mass', rho*A)

            # ops.element('elasticBeamColumn', i+1, i+1, i+2, A, E, G, J, Iy, Iz, 1, '-mass', rho*A, '-cMass')
            # ops.eleLoad('-ele', i+1, '-type','-beamUniform', dis_load_twr1[1], dis_load_twr1[2], dis_load_twr1[0])

            if i == 0:
                ops.element('dispBeamColumn', i+nelem+1, i+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                # ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
            else:
                ops.element('dispBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                # ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
                # print(f'node1:{i+nelem+1},  node2:{i+nelem+2}')
        
        # add truss element representing cable
        ops.uniaxialMaterial('Elastic', 200, E)
        ops.element('corotTruss', 1000, nelem+1, 2*nelem+1, 0.25, 200, '-rho', rho*1e-12)
        ops.element('corotTruss', 1001, (nelem+3)//2, (nelem+3)//2+nelem, 0.25, 200, '-rho', rho*1e-12)
        
        # addition mass and inertia from naccele and rotor
        mass_trans = 3.5e5
        inertia1 = 4.37e7
        inertia2 = 2.353e7
        inertia3 = 2.542e7

        ops.mass(nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        ops.mass(2*nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        
        time = np.linspace(0, 100, 1000)  # Example time vector
        displacement_x = 0.1 * np.sin(2 * np.pi * 0.1 * time)  # Example sinusoidal displacement in x-direction

        ops.timeSeries('Path', 1, '-dt', time[1] - time[0], '-values', *displacement_x)
        ops.pattern('MultipleSupport', 1)
        # load patern
        # ops.timeSeries('Constant', 1)
        # ops.pattern('Plain', 1, 1)
        

        # Define the ground motion at the fixed support
        # 'Plain' indicates a direct application of the time series
        # The node tag is the fixed support node (node 1)
        # The DOF is the direction of the imposed displacement (1 for x-direction)
        # The time series tag is the one we defined (tag 1)
        ops.groundMotion(1, 'Plain', '-disp', 1)
        ops.imposedMotion(1, 1, 1)  # Apply groundMotion tag 1 to node 1, DOF 1

        # --- Analysis ---
        # Define the analysis parameters
        dt = time[1] - time[0]
        num_steps = len(time)

        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')  # For linear analysis, use 'Linear'
        ops.test('NormDispIncr', 1.0e-6, 10)
        ops.algorithm('Linear')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        # Record results (optional)
        ops.recorder('Node', '-file', 'displacement_response.out', '-time', '-node', 3, '-dof', 1, 2, 'disp')
        ops.recorder('Element', '-file', 'force_response.out', '-time', '-ele', 1, 'localForce')
        # ops.system('BandGeneral')
        # ops.numberer('RCM')
        # ops.constraints('Plain')
        # ops.test('NormDispIncr', 1e-6, 10)
        # ops.algorithm('ModifiedNewton')
        # ops.integrator('LoadControl', 1.0)
        # ops.analysis('Static')

        # rotor load
        # ops.load(nelem+1, *f_aero_twr_top[0])
        # ops.load(2*nelem+1, *f_aero_twr_top[1])

        ops.analyze(num_steps, dt)
        
        import opsvis as opsv    

        ops.analyze(1)

        def calc_stress(ele=0, secType='circular', n=200, plot_stress=True):
            
            Fx, Fy, Fz, Mx, My, Mz = ops.eleResponse(ele+1, 'localForce')[:6]

            A, Iy, Iz, J, d, t = sec_prop[ele,:]

            # if secType == 'circular':
            #     Sz = Sy = 2 * ((d-t)/2)**2 * t
            #     y = z = 0.5*d
            # elif secType == 'ellipse':
            #     NotImplementedError()
            r = 0.5*d

            def compute_vom_mises(r, theta):
                """
                Compute the second deviatoric stress invariant J2 from a stress tensor 
                given in Voigt notation (assuming shear components are not multiplied by 2).

                Parameters:
                    sigma_voigt: array-like of length 6
                        The stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

                Returns:
                    J2: float
                        The second invariant of the deviatoric stress tensor
                """
                y = r*np.cos(theta)
                z = r*np.sin(theta)

                sigma_xx = Fx/A + My*z/Iy + Mz*y/Iz
                sigma_yy = 0
                sigma_zz = 0  

                sigma_xy = Fy/(0.25*np.pi*d**2)/2/t
                sigma_xz = Fz/(0.25*np.pi*d**2)/2/t
                sigma_yz = Mx*r/J
                
                s = np.zeros(6)
                mean_stress = np.mean([sigma_xx, sigma_yy, sigma_zz])
                s[:3] = np.array([sigma_xx, sigma_yy, sigma_zz]) - mean_stress
                s[3:] = np.array([sigma_xy, sigma_xz, sigma_yz])

                J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2 +
                            2 * (s[3]**2 + s[4]**2 + s[5]**2))
                
                von_mises = np.sqrt(3*J2)
                
                return von_mises

            alpha = np.arctan2(My/Iz, Mz/Iy)
            # alpha = np.arctan2(Mz/Iy, My/Iz)

            max_von_mises = compute_vom_mises(r, alpha)

            if plot_stress:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import cm
                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection

                theta = np.linspace(0, np.pi*2, n, endpoint=True)

                von_mises = [compute_vom_mises(r, theta[i]) for i in range(n)]

                factor = 10
                pts_out = np.array([r*np.cos(theta), r*np.sin(theta)])
                pts_in = np.array([(r-t*factor)*np.cos(theta), (r-t*factor)*np.sin(theta)])

                patches = []
                colors = []

                for i in range(n-1):
                    quad = [pts_in[:,   i].tolist(), 
                            pts_in[:, i+1].tolist(), 
                            pts_out[:,i+1].tolist(), 
                            pts_out[:,  i].tolist()]

                    polygon = Polygon(quad, closed=True)
                    patches.append(polygon)
                    colors.append(von_mises[i])

                fig, ax = plt.subplots(figsize=(6,6))
                p = PatchCollection(patches, cmap='coolwarm', edgecolor='k', alpha=0.9, linewidth=0.0)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax, label='Shear Stress')

                ax.set_aspect('equal')
                ax.set_xlim(-1.2*r, 1.2*r)
                ax.set_ylim(-1.2*r, 1.2*r)
                ax.set_title('Thin-Walled Circular Section with Shear Stress Distribution')

            
            return max_von_mises
        
        # von_mises = np.zeros(nelem)
        # for i in range(nelem):
        #     von_mises[i] = calc_stress(i, plot_stress=False)

        # max von_mises in twr bottom
        von_mises_bot = calc_stress(0, plot_stress=False)
        
        # eigen anlysis
        num_modes = 12
        eigenvals = ops.eigen(num_modes)
        freqs = [np.sqrt(lam)/(2*np.pi) for lam in eigenvals]

        print("\nFrequency(Hz):")
        for i, f in enumerate(freqs):
            print(f"Mode {i+1}: {f:.4f} Hz")
        
        # Tn = [1/0.3656]
        # Sa = [11]
        
        ops.modalProperties("-print", "-file", "ModalReport.txt", "-unorm")

        # filename = 'ele_1_sec_1.txt'
        # ops.recorder('Element', '-xml', filename, '-closeOnWrite', '-precision', 4, '-ele', 1, 'localForce')
        # ops.recorder('Node', '-file', 'ele_1_sec_2.txt', '-closeOnWrite', '-precision', 4, '-node', 1, 'reaction')
        
        
        for i, (T, S) in enumerate(zip(Tn, Sa)):
            # ops.wipeAnalysis()
            ops.responseSpectrumAnalysis(1, '-Tn', T, '-Sa', S)
            force = np.array(ops.eleResponse(1, 'localForce')[:6])
            print(force[0])

        # for node in ops.getNodeTags():
        #     disps = np.array(ops.nodeDisp(node))
        #     ops.reactions() # Must call this command before using nodeReaction() command.
        #     reactions = np.array(ops.nodeReaction(node))
        #     force = np.array(ops.eleResponse(node, 'localForce')[:6])
        a = 1
        mod = np.zeros([num_modes, 2*nelem+1, 6])
        disp = np.zeros([2*nelem+1, 6])
        cable_tension = np.zeros(2)

        for i in range(2*nelem+1):
            for j in range(num_modes):
                mod[j,i,:] = ops.nodeEigenvector(i+1, j+1)
                # mod[j,i,3:] = mod[j,i,3:] * 180/np.pi
            
            disp[i] = ops.nodeDisp(i+1)
            # disp[i,3:] = disp[i,3:] * 180/np.pi
        
        # tmp = ops.eleForce(1000)
        cable_tension[0] = ops.eleForce(1000, 2)
        cable_tension[1] = ops.eleForce(1001, 2)
        

        def plot_static_disp(factor=1e4):

            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection


            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            l = np.linalg.norm(rTwr1) * np.linspace(0,1, nelem+1, endpoint=True)

            colors = cm.inferno(np.linspace(0.15, 0.7, 6))

            lines = [disp[:nelem+1, i]*factor for i in range(6)]

            for i in range(6):
                ax.plot(np.ones_like(l)*(i+1) , l , lines[i], color=colors[i], label=f'mod{i}', linewidth=2.0, linestyle='--')

                verts = [list(zip(np.ones_like(l)*(i+1), l, np.full_like(l, lines[i])))] 
                verts[0].append([i+1,l[-1],0]) 
                verts[0].append([i+1,0,0])
                poly = Poly3DCollection(verts, facecolors=colors[i], alpha=0.15)  # 创建多边形集合

                ax.add_collection3d(poly, zs=-1, zdir='y')  # 添加到 3D 图中
                ax.invert_xaxis()
                ax.grid()
                ax.legend()
                ax.set_zlabel(f'disp*{factor:.0e}')
                ax.set_xlabel(f'Mods')
                ax.set_ylabel(f'length')

            return ax


        def plot_eigen_disp(nf=0, factor=1e4):

            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection


            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            l = np.linalg.norm(rTwr1) * np.linspace(0,1, nelem+1, endpoint=True)

            colors = cm.inferno(np.linspace(0.15, 0.7, 6))

            lines = [np.abs(mod[nf, :nelem+1, i])*factor for i in range(6)]

            for i in range(6):
                ax.plot(np.ones_like(l)*(i+1) , l , lines[i], color=colors[i], label=f'mod{i}', linewidth=2.0, linestyle='--')

                verts = [list(zip(np.ones_like(l)*(i+1), l, np.full_like(l, lines[i])))] 
                verts[0].append([i+1,l[-1],0]) 
                verts[0].append([i+1,0,0])
                poly = Poly3DCollection(verts, facecolors=colors[i], alpha=0.15)  # 创建多边形集合

                ax.add_collection3d(poly, zs=-1, zdir='y')  # 添加到 3D 图中
                ax.invert_xaxis()
                ax.grid()
                ax.legend()
                ax.set_zlabel(f'disp*{factor:.0e}')
                ax.set_xlabel(f'Mods')
                ax.set_ylabel(f'length')

            return ax


        def plot_static_3D(factor=[1e4,1e3,1e4], plot_rotor=True, plot_tower=True, plot_cable=True):

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            if plot_tower:          
                mem0.plot(ax)
                mem1.plot(ax)
            else:
                ax.plot(r_node_twr1[:,0],r_node_twr1[:,1],r_node_twr1[:,2], linewidth=5.0, color='grey')
                ax.plot(r_node_twr2[:,0],r_node_twr2[:,1],r_node_twr2[:,2], linewidth=5.0, color='grey')

            if plot_cable:
                ax.plot([r_node_twr1[-1,0],r_node_twr2[-1,0]],
                        [r_node_twr1[-1,1],r_node_twr2[-1,1]],
                        [r_node_twr2[-1,2],r_node_twr1[-1,2]], 
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
        
                ax.plot([r_node_twr1[(nelem+1)//2,0],r_node_twr2[(nelem+1)//2,0]],
                        [r_node_twr1[(nelem+1)//2,1],r_node_twr2[(nelem+1)//2,1]],
                        [r_node_twr2[(nelem+1)//2,2],r_node_twr1[(nelem+1)//2,2]],
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
            
            twr1_mod = r_node_twr1[:,:3]+disp[:nelem+1, :3] * factor
            twr2_mod = r_node_twr2[:,:3]+np.vstack([disp[0, :3], disp[nelem+1:, :3]]) * factor

            ax.plot(twr1_mod[:,0],
                    twr1_mod[:,1],
                    twr1_mod[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
        
            ax.plot(twr2_mod[:,0],
                    twr2_mod[:,1],
                    twr2_mod[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
            
            if plot_rotor:
                fowt.rotorList[0].plot(ax, airfoils=False, draw_circle=True)
                fowt.rotorList[1].plot(ax, airfoils=False, draw_circle=True)

            '''
            plot cable after deformation
            # ax.plot([twr1_mod1[-1,0],twr2_mod1[-1,0]],
            #         [twr1_mod1[-1,1],twr2_mod1[-1,1]],
            #         [twr1_mod1[-1,2],twr2_mod1[-1,2]], 
            #         linewidth=2.0, 
            #         color='r', 
            #         linestyle='--', 
            #         alpha=0.8)

            # ax.plot([twr1_mod1[(nelem+1)//2,0],twr2_mod1[(nelem+1)//2,0]],
            #         [twr1_mod1[(nelem+1)//2,1],twr2_mod1[(nelem+1)//2,1]],
            #         [twr1_mod1[(nelem+1)//2,2],twr2_mod1[(nelem+1)//2,2]],
            #           linewidth=2.0, 
            #           color='r', 
            #           linestyle='--', 
            #           alpha=0.8)

            '''

            '''
            # nn = 10
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn] * 0.5


            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')


            # nn = 12
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn]*0.5

            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')
            '''
            
            def set_axes_equal(ax):
                '''Set 3D plot axes to equal scale'''
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                max_range = max(x_range, y_range, z_range)

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
                ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
                ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

            set_axes_equal(ax)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Structure View: Two Towers + Cable')
            ax.view_init(elev=20, azim=45)
            ax.grid(True)

            return ax


        def plot_eigen_3D(nf=0, factor=[1e4,1e3,1e4], plot_rotor=True, plot_tower=True, plot_cable=True):

            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            if plot_tower:          
                mem0.plot(ax)
                mem1.plot(ax)
            else:
                ax.plot(r_node_twr1[:,0],r_node_twr1[:,1],r_node_twr1[:,2], linewidth=5.0, color='grey')
                ax.plot(r_node_twr2[:,0],r_node_twr2[:,1],r_node_twr2[:,2], linewidth=5.0, color='grey')

            if plot_cable:
                ax.plot([r_node_twr1[-1,0],r_node_twr2[-1,0]],
                        [r_node_twr1[-1,1],r_node_twr2[-1,1]],
                        [r_node_twr2[-1,2],r_node_twr1[-1,2]], 
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
        
                ax.plot([r_node_twr1[(nelem+1)//2,0],r_node_twr2[(nelem+1)//2,0]],
                        [r_node_twr1[(nelem+1)//2,1],r_node_twr2[(nelem+1)//2,1]],
                        [r_node_twr2[(nelem+1)//2,2],r_node_twr1[(nelem+1)//2,2]],
                        linewidth=2.0, 
                        color='black', 
                        linestyle='--', 
                        alpha=0.6)
            
            if nf > 1 & nf < 6:
                factor[0] *= 100
            twr1_mod1 = r_node_twr1[:,:3]+mod[nf, :nelem+1, :3] * factor
            twr2_mod1 = r_node_twr2[:,:3]+np.vstack([mod[nf, 0, :3], mod[nf, nelem+1:, :3]]) * factor

            ax.plot(twr1_mod1[:,0],
                    twr1_mod1[:,1],
                    twr1_mod1[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
        
            ax.plot(twr2_mod1[:,0],
                    twr2_mod1[:,1],
                    twr2_mod1[:,2], 
                    linewidth=4.0, 
                    color='r', 
                    linestyle='--', 
                    alpha=0.8)
            
            if plot_rotor:
                fowt.rotorList[0].plot(ax, airfoils=False, draw_circle=True)
                fowt.rotorList[1].plot(ax, airfoils=False, draw_circle=True)

            '''
            plot cable after deformation
            # ax.plot([twr1_mod1[-1,0],twr2_mod1[-1,0]],
            #         [twr1_mod1[-1,1],twr2_mod1[-1,1]],
            #         [twr1_mod1[-1,2],twr2_mod1[-1,2]], 
            #         linewidth=2.0, 
            #         color='r', 
            #         linestyle='--', 
            #         alpha=0.8)

            # ax.plot([twr1_mod1[(nelem+1)//2,0],twr2_mod1[(nelem+1)//2,0]],
            #         [twr1_mod1[(nelem+1)//2,1],twr2_mod1[(nelem+1)//2,1]],
            #         [twr1_mod1[(nelem+1)//2,2],twr2_mod1[(nelem+1)//2,2]],
            #           linewidth=2.0, 
            #           color='r', 
            #           linestyle='--', 
            #           alpha=0.8)

            '''

            '''
            # nn = 10
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn] * 0.5


            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')


            # nn = 12
            # theta = np.linspace(0,2*np.pi, 40, endpoint=True)
            # r = 0.5*d_node[nn]*0.5

            # pts = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]) # npts , 3
            # pts = np.matmul(mem0.R, pts) + r_node_twr1[nn,:][:,None]
            # ax.plot(pts[0], pts[1], pts[2], linestyle=':', color='grey')
            '''
            
            def set_axes_equal(ax):
                '''Set 3D plot axes to equal scale'''
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                max_range = max(x_range, y_range, z_range)

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
                ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
                ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

            set_axes_equal(ax)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Structure View: Two Towers + Cable')
            ax.view_init(elev=20, azim=45)
            ax.grid(True)

        # plt.tight_layout()
        # plt.show()

            return ax

        # ax = plot_eigen_disp(nf=2, factor=1e4)
        # ax = plot_static_disp(factor=1)
        # ax = plot_eigen_3D(nf=5, factor=[1e5,1e4,1e4])
        # ax = plot_static_3D(factor=[2e1,1e3,1e2])
        # plt.tight_layout()
        # plt.show()

        if plot:
            ax0 = plot_eigen_3D(0, factor=[1e4,1e4,1e4])
            ax1 = plot_eigen_3D(1, factor=[1e4,1e4,1e4])
            ax2 = plot_eigen_3D(2, factor=[1e2,1e4,1e4])
            ax3 = plot_eigen_3D(3, factor=[1e4,1e4,1e4])
            ax4 = plot_eigen_3D(4, factor=[1e4,1e4,1e4])
            ax5 = plot_eigen_3D(5, factor=[1e4,1e4,1e4])
            ax6 = plot_static_3D(factor=[2e1,1e3,1e2])

            return von_mises_bot, cable_tension[0], freqs, [ax0,ax1,ax2,ax3,ax4,ax5,ax6]
        else:

            return von_mises_bot, cable_tension[0], freqs
        



def solveTwrCombinationOpenseesCoupled(fowt:FOWT, case, nelem=20, r6=np.zeros(6)):
        '''Calculate and output tower bending moment without considering paltform (considering FOWT in a psuedo
        'static' condition) in order to optimize Nezzy2-like twin-rotor FOWT.

        Input:
            fowt (raft_fowt.FOWT): a deepcopy of FOWT instance
            case (case)
            pitch (rad)

        Output:
            Mbase_list (np.array([3, nrotors]))

        '''     
        import openseespy.opensees as ops

        fowt.setPosition(r6)
        # fowt.calcTurbineConstants(case, ptfm_pitch=r6[4])
        # fowt.calcTowerAeroLoads(case)

        f_aero0 = np.zeros([fowt.ntowers, 6])
        f_aero_twr_top = np.zeros([fowt.ntowers, 6])

        E = 210*1e9        # Modulus of elasticity (Pa)
        G = 80.8*1e9       # Shear modulus of elasticity (Pa)
        nu = 0.3           # Poisson's ratio
        rho = 8500         # Density (kg/m**3)

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)


        for ir, rot in enumerate(fowt.rotorList):

            mem = fowt.memberList[fowt.nplatmems + ir]
            # mem.setPosition(r6=[0,0,0,0,fowt.r6[4],0])

            f_aero0[ir,:], _, _, _ = rot.calcAero(case, current=False)

            # counter-clockwise rotating rotor
            # An approximate method (needs update)
            if ir == 1:
                  f_aero0[ir,1] = - f_aero0[ir,1]
                  f_aero0[ir,3] = - f_aero0[ir,3]
                  f_aero0[ir,5] = - f_aero0[ir,5]

            f_aero_twr_top[ir] = transformForce(f_aero0[ir,:], offset=(rot.r_hub_rel-mem.rB))
            f_aero_twr_top[ir] += transformForce([0,0,-rot.mRNA * fowt.g,0,0,0], offset=(rot.r_CG-mem.rB))

        mem0 = fowt.memberList[fowt.nplatmems]
        mem1 = fowt.memberList[fowt.nplatmems+1]

        rTwr1 = mem0.rB - mem0.rA
        rTwr2 = mem1.rB - mem1.rA

        r_node_twr1 = np.zeros([nelem+1,3])
        r_node_twr2 = np.zeros([nelem+1,3])

        d_node = np.linspace(mem0.d[0], mem0.d[-1], nelem+1, endpoint=True)
        t_node = np.linspace(mem0.t[0], mem0.t[-1], nelem+1, endpoint=True)

        # ops add nodes
        for i, n in enumerate(np.linspace(0,1,nelem+1, endpoint=True)):
            r_node_twr1[i] = mem0.rA + n * rTwr1
            r_node_twr2[i] = mem1.rA + n * rTwr2
        
        r_node = np.concatenate([r_node_twr1, r_node_twr2[1:]], axis=0)

        for inode, r in enumerate(r_node):    
            
            ops.node(inode+1, r[0], r[1], r[2])

        # ops fix bottom
        ops.fix(1, 1,1,1, 1,1,1)
            
        # ops add transform
        axis1 = np.cross(-mem0.p2, rTwr1)
        axis1 = axis1 / np.linalg.norm(axis1)
        R1 = rotationMatrix(rTwr1, ref=mem0.p2)
        
        axis2 = np.cross(-mem1.p2, rTwr2)
        axis2 = axis2 / np.linalg.norm(axis2)
        R2 = rotationMatrix(rTwr2, ref=mem1.p2)
        
        axis3 = [0, 0, 1]

        ops.geomTransf('Corotational', 1, *axis1)
        ops.geomTransf('Corotational', 2, *axis2)  
        ops.geomTransf('Corotational', 3, *axis3)
        

        # load patern
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)

        sec_prop = np.zeros([nelem, 6])
        # ops add elements
        for i in range(nelem):
            A0, Iy0, Iz0, J0 = section_property(d_node[i], t_node[i])
            A1, Iy1, Iz1, J1 = section_property(d_node[i+1], t_node[i+1])

            A, Iy, Iz, J = np.array([A0+A1, Iy0+Iy1, Iz0+Iz1, J0+J1])/2
            sec_prop[i] = [A0, Iy0, Iz0, J0, d_node[i], t_node[i]]

            ops.section('Elastic', i+1, E,A, Iz, Iy, G, J)
            ops.beamIntegration('Lobatto', i+1, i+1, 5)

            g = -9.81
            Cd = 1.0
            hHub = fowt.rotorList[0].hHub
            v = case['wind_speed'] * (r_node[i, 2]/hHub)**fowt.shearExp_air

            v1 = R1.T @ np.array([v,0,0])
            v2 = R2.T @ np.array([v,0,0])

            tmp = R1.T @  np.array([0,0,-1])

            dis_load_twr1 = R1.T @ (np.array([0, 0, g*rho*A])) + np.array([0,-1*0.5*1.225*v1[1]**2*d_node[i]*Cd,0])
            dis_load_twr2 = R2.T @ (np.array([0, 0, g*rho*A])) + np.array([0,0.5*1.225*v2[1]**2*d_node[i]*Cd,0])
            
            ops.element('dispBeamColumn', i+1, i+1, i+2, 1, i+1, '-cMass', '-mass', rho*A)

            # ops.element('elasticBeamColumn', i+1, i+1, i+2, A, E, G, J, Iy, Iz, 1, '-mass', rho*A, '-cMass')
            ops.eleLoad('-ele', i+1, '-type','-beamUniform', dis_load_twr1[1], dis_load_twr1[2], dis_load_twr1[0])

            if i == 0:
                ops.element('dispBeamColumn', i+nelem+1, i+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
            else:
                ops.element('dispBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, 2, i+1, '-cMass', '-mass', rho*A)
                # ops.element('elasticBeamColumn', i+nelem+1, i+nelem+1, i+nelem+2, A, E, G, J, Iy, Iz, 2, '-mass', rho*A, '-cMass')

                ops.eleLoad('-ele', i+nelem+1, '-type','-beamUniform', dis_load_twr2[1], dis_load_twr2[2], dis_load_twr2[0])
                # print(f'node1:{i+nelem+1},  node2:{i+nelem+2}')
        
        # add truss element representing cable
        A1 = 0.25*np.pi*0.30**2
        A2 = 0.25*np.pi*0.25**2
        ops.uniaxialMaterial('Elastic', 200, E)
        ops.element('corotTruss', 1000, nelem+1, 2*nelem+1, A1, 200, '-rho', rho*1e-12)
        ops.element('corotTruss', 1001, (nelem+3)//2, (nelem+3)//2+nelem, A2, 200, '-rho', rho*1e-12)
        
        # addition mass and inertia from naccele and rotor
        mass_trans = 3.5e5
        inertia1 = 4.37e7
        inertia2 = 2.353e7
        inertia3 = 2.542e7

        ops.mass(nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        ops.mass(2*nelem+1, mass_trans, mass_trans, mass_trans, inertia1, inertia2, inertia3)
        
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.test('NormDispIncr', 1e-9, 100)
        ops.algorithm('KrylovNewton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')

        # rotor load
        ops.load(nelem+1, *f_aero_twr_top[0])
        ops.load(2*nelem+1, *f_aero_twr_top[1])

        import opsvis as opsv    

        ops.analyze(1)
        twr_top_disp_x = ops.nodeDisp(nelem+1)[0]
        print(f'node Dispx of top = {twr_top_disp_x:.3e}')

        twr_top_disp_y = ops.nodeDisp(nelem+1)[1]
        print(f'node Dispx of top = {twr_top_disp_y:.3e}')

        
        sigma_cable1 = ops.eleResponse(1000, 'forces')[1] / A1
        sigma_cable2 = ops.eleResponse(1001, 'forces')[1] / A2
        print(f'cable1 stress = {sigma_cable1/1e6:.3e} Mpa')
        print(f'cable2 stress = {sigma_cable2/1e6:.3e} Mpa')
        # print(f'node Dispx of top = {ops.nodeDisp(2*nelem+1)[0]:.3e}')

        # F_TB = -1 * (ops.eleResponse(1, 'forces')[:6] + ops.eleResponse(1+nelem, 'forces')[:6])
        F_TB_1 = -1 * np.array(ops.eleResponse(1, 'forces')[:6])
        F_TB_2 = -1 * np.array(ops.eleResponse(nelem+1, 'forces')[:6])

        F_TB = F_TB_1 + F_TB_2

        def calc_stress(ele=0, secType='circular', n=200, plot_stress=True):
            
            Fx, Fy, Fz, Mx, My, Mz = ops.eleResponse(ele+1, 'localForce')[:6]

            A, Iy, Iz, J, d, t = sec_prop[ele,:]

            # if secType == 'circular':
            #     Sz = Sy = 2 * ((d-t)/2)**2 * t
            #     y = z = 0.5*d
            # elif secType == 'ellipse':
            #     NotImplementedError()
            r = 0.5*d

            def compute_vom_mises(r, theta):
                """
                Compute the second deviatoric stress invariant J2 from a stress tensor 
                given in Voigt notation (assuming shear components are not multiplied by 2).

                Parameters:
                    sigma_voigt: array-like of length 6
                        The stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]

                Returns:
                    J2: float
                        The second invariant of the deviatoric stress tensor
                """
                y = r*np.cos(theta)
                z = r*np.sin(theta)

                sigma_xx = Fx/A - My*z/Iy + Mz*y/Iz
                sigma_yy = 0
                sigma_zz = 0  

                sigma_xy = Fy/(0.25*np.pi*d**2)/2/t
                sigma_xz = Fz/(0.25*np.pi*d**2)/2/t
                sigma_yz = Mx*r/J
                
                s = np.zeros(6)
                mean_stress = np.mean([sigma_xx, sigma_yy, sigma_zz])
                s[:3] = np.array([sigma_xx, sigma_yy, sigma_zz]) - mean_stress
                s[3:] = np.array([sigma_xy, sigma_xz, sigma_yz])

                J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2 +
                            2 * (s[3]**2 + s[4]**2 + s[5]**2))
                
                von_mises = np.sqrt(3*J2)
                
                return von_mises

            alpha = np.arctan2(-My/Iz, Mz/Iy)
            # alpha = np.arctan2(Mz/Iy, My/Iz)

            max_von_mises = compute_vom_mises(r, alpha)

            if plot_stress:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib import cm
                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection

                theta = np.linspace(0, np.pi*2, n, endpoint=True)

                von_mises = [compute_vom_mises(r, theta[i]) for i in range(n)]

                factor = 10
                pts_out = np.array([r*np.cos(theta), r*np.sin(theta)])
                pts_in = np.array([(r-t*factor)*np.cos(theta), (r-t*factor)*np.sin(theta)])

                patches = []
                colors = []

                for i in range(n-1):
                    quad = [pts_in[:,   i].tolist(), 
                            pts_in[:, i+1].tolist(), 
                            pts_out[:,i+1].tolist(), 
                            pts_out[:,  i].tolist()]

                    polygon = Polygon(quad, closed=True)
                    patches.append(polygon)
                    colors.append(von_mises[i])

                fig, ax = plt.subplots(figsize=(6,6))
                p = PatchCollection(patches, cmap='coolwarm', edgecolor='k', alpha=0.9, linewidth=0.0)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax, label='Shear Stress')

                ax.set_aspect('equal')
                ax.set_xlim(-1.2*r, 1.2*r)
                ax.set_ylim(-1.2*r, 1.2*r)
                ax.set_title('Thin-Walled Circular Section with Shear Stress Distribution')

            
            return max_von_mises
        

        von_mises_TB = calc_stress(0, plot_stress=False)
        # plt.show()
        # eigen anlysis
        num_modes = 6
        eigenvals = ops.eigen(num_modes)
        freqs = [np.sqrt(lam)/(2*np.pi) for lam in eigenvals]

        print("\nFrequency(Hz):")
        for i, f in enumerate(freqs):
            print(f"Mode {i+1}: {f:.4f} Hz")
        
        # mod = np.zeros([num_modes, 2*nelem+1, 6])
        # disp = np.zeros([2*nelem+1, 6])
        # cable_tension = np.zeros(2)

        # for i in range(2*nelem+1):
        #     for j in range(num_modes):
        #         mod[j,i,:] = ops.nodeEigenvector(i+1, j+1)
        #         # mod[j,i,3:] = mod[j,i,3:] * 180/np.pi
            
        #     disp[i] = ops.nodeDisp(i+1)
        #     # disp[i,3:] = disp[i,3:] * 180/np.pi
        
        # # tmp = ops.eleForce(1000)
        # cable_tension[0] = ops.eleForce(1000, 2)
        # cable_tension[1] = ops.eleForce(1001, 2)

       
        return F_TB, von_mises_TB, twr_top_disp_x, freqs