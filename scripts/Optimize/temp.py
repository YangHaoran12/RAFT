import openmdao.api as om
import raft
import numpy as np
import pickle, os
import copy
from itertools import compress
from wisdem.inputs import write_yaml, simple_types
import yaml
from raft.raft_rotor import raft_dir
from raft.raft_fowt import FOWT
import os.path as osp
from typing import Iterable
from raft.opt_helpers import calcStaticTwrBendingMoment

DEBUG_OMDAO = False  # use within WEIS, test file generated using examples/15_RAFT_Studies/weis_driver_raft_opt.py

ndim = 3
ndof = 6
tower_npts = 11

fname_design = os.path.join(raft_dir, "designs/OC4semi_twin.yaml")

min_freq = 0.05 * np.pi*2
max_freq = 0.2 * np.pi*2

z_offset = 2.4

draft = -20



with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    design['turbine']['rho_air' ] = design['site']['rho_air']
    design['turbine']['mu_air'  ] = design['site']['mu_air']
    design['turbine']['shearExp_air'] = design['site']['shearExp']
    design['turbine']['shearExp_water'] = design['site']['shearExp']

    # zero the nacelle velocity feedback gain since there seems to be a discrepancy with its definition
    design['turbine']['pitch_control']['Fl_Kp'] = 0.0

case = dict(zip(design['cases']['keys'], design['cases']['data'][0]))

# turbine_design = design['turbine']
Rtip = float(design['turbine']['blade']['Rtip'])
Zhub = float(design['turbine']['Zhub'])

stations = [ 10, 17.76, 25.52, 33.28, 41.04, 48.8, 56.56, 64.32, 72.08, 79.84, 87.6 ]    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
d        = [ 6.5, 6.237, 5.974, 5.711, 5.448, 5.185, 4.922, 4.659, 4.396, 4.133, 3.870 ]    # [m]    diameters if circular or side lengths if rectangular (can be pairs)
t        = [ 0.027, 0.0262, 0.0254, 0.0246, 0.0238, 0.023, 0.0222, 0.0214, 0.0206, 0.0198, 0.0190 ]                     # [m]    wall thicknesses (scalar or list of same length as stations)

from scipy.interpolate import interp1d

l_stations_normalized = (np.array(stations)-stations[0])/(stations[-1]-stations[0])

d_interp = interp1d(l_stations_normalized, d, kind='linear', fill_value='extrapolate')
t_interp = interp1d(l_stations_normalized, t, kind='linear', fill_value='extrapolate')


class Rotors(om.ExplicitComponent):
    """
    RAFT OpenMDAO Wrapper for twin rotor configuration

    """

    def initialize(self):
        '''
        Inintialize turbine configurations as an 'om.ExplicitComponent.options'
        '''
        self.options.declare('turbine_options', default=design['turbine'] , types=dict)
        # self.options.declare('platform_options', default=design['platform'] , types=dict)


    def setup(self):

        turbine_opt = self.options['turbine_options']      

        # add input
        self.add_input('turbine_L_rotors',   val=0.0, units='m', shape=1, desc='Length(y offset)between twin rotors')
        self.add_input('turbine_zRNA',       val=0.0, units='m', shape=1, desc='z coordinate of rotor hub of each rotor') 
        self.add_input('turbine_tower_rA_z', val=0.0, units='m', shape=1, desc='z coordinate of tower base of each rotor')

        # add output
        self.add_output('max_tower_stress', val=0, shape=1, desc = 'Maximum tower base moment', units='Pa')
        self.add_output('yRNA', val=0.0,  shape=1, desc = 'rRNA.y of rotor 1', units='m')
        self.add_output('hHub', val=0.0,  shape=1, desc = 'hHub of each rotor', units='m')
        self.add_output('hTower', val=0.0, shape=1, desc = 'h of each tower', units='m')
        
        self.add_output('turbine_tower1_rB', val=np.zeros(ndim), units='m', desc='tower1 End B coordinates')
        self.add_output('turbine_tower2_rB', val=np.zeros(ndim), units='m', desc='tower2 End B coordinates')

        self.add_output('con1', val=0.0, units='m', desc='constraint1')
        
        # stations 
        # self.add_output('turbine_tower_stations', val=np.zeros(tower_npts), units=None, desc='Location of stations along axis, will be normalized along rA to rB')

        # diameter or side length
        if turbine_opt['tower'][0]['shape'] == 'circ':
            self.add_output('turbine_tower_d', val=np.zeros(tower_npts), units='m', desc='Diameters if circular')
    
        elif turbine_opt['tower'][0]['shape'] == 'rect' or 'ellipse':
            self.add_output('turbine_tower_sl', val=np.zeros(tower_npts, 2), units='m', desc='side lengths if rectangular or ellipse')

        # thickness
        self.add_output('turbine_tower_t', val=np.zeros(tower_npts), units='m', desc='Wall thicknesses at station locations')
    
    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    # compute 
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turbine_opt = self.options['turbine_options']

        outputs['yRNA'] = inputs['turbine_L_rotors'] * 0.5
        outputs['hHub'] =  inputs['turbine_zRNA']
        outputs['hTower'] = inputs['turbine_zRNA'] - z_offset

        # print(inputs['turbine_L_rotors']/2)
        outputs['turbine_tower1_rB'] = np.array([0,  inputs['turbine_L_rotors'].item()*0.5, inputs['turbine_zRNA'].item()])
        outputs['turbine_tower2_rB'] = np.array([0, -inputs['turbine_L_rotors'].item()*0.5, inputs['turbine_zRNA'].item()])
        
        outputs['con1'] = outputs['hTower'] - inputs['turbine_L_rotors']*0.5 - inputs['turbine_tower_rA_z']

        # outputs['turbine_tower_stations'] = np.linspace(0, 1, tower_npts, endpoint=True)

        if turbine_opt['tower'][0]['shape'] == 'circ':
            outputs['turbine_tower_d'] = d_interp(np.linspace(0, 1, tower_npts, endpoint=True))

        elif turbine_opt['tower'][0]['shape'] == 'rect' or 'ellipse':

            raise NotImplementedError()
            # outputs['turbine_tower_t'][:,0] = d_interp(outputs['turbine_tower_stations'])

        outputs['turbine_tower_t'] = t_interp(np.linspace(0, 1, tower_npts, endpoint=True))
        
        # new_turbine_design = copy.deepcopy(design)
        
        turbine_opt['rRNA'][0][1] =  outputs['yRNA'].item()
        turbine_opt['rRNA'][1][1] = -outputs['yRNA'].item()
        turbine_opt
        turbine_opt['tower'][0]['rB'][1] =  outputs['yRNA'].item()
        turbine_opt['tower'][1]['rB'][1] = -outputs['yRNA'].item()

        turbine_opt['hHub']              =  outputs['hHub'].item()

        turbine_opt['tower'][0]['rB'][2]   =  outputs['hTower'].item() - z_offset
        turbine_opt['tower'][1]['rB'][2]   =  outputs['hTower'].item() - z_offset

        turbine_opt['tower'][0]['rA'][2]   =  inputs['turbine_tower_rA_z'].item()
        turbine_opt['tower'][1]['rA'][2]   =  inputs['turbine_tower_rA_z'].item()

        # new_turbine_design['turbine']['rRNA'][0][1] =  outputs['yRNA'].item()
        # new_turbine_design['turbine']['rRNA'][1][1] = -outputs['yRNA'].item()
        
        # new_turbine_design['turbine']['tower'][0]['rB'][1] = outputs['yRNA'].item()
        # new_turbine_design['turbine']['tower'][1]['rB'][1] = outputs['yRNA'].item()

        # new_turbine_design['turbine']['hHub']              =  outputs['hHub'].item()

        # new_turbine_design['turbine']['tower'][0]['rB'][2]   =  outputs['hTower'].item() - z_offset
        # new_turbine_design['turbine']['tower'][1]['rB'][2]   =  outputs['hTower'].item() - z_offset

        # new_turbine_design['turbine']['tower'][0]['rA'][2]   =  inputs['turbine_tower_rA_z'].item()
        # new_turbine_design['turbine']['tower'][1]['rA'][2]   =  inputs['turbine_tower_rA_z'].item()




        fowt = FOWT(design, np.linspace(min_freq, max_freq, 81), None, depth=200, x_ref=0, y_ref=0, heading_adjust=0)
        fowt.setPosition([0,0,0,0,0,0])
        _, stress = calcStaticTwrBendingMoment(fowt.copy(), case)

        outputs['max_tower_stress'] = np.max(np.abs(stress))

        return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)
    
    # def final_setup(self):

    #     return self.options['turbine_options']


class Platform(om.ExplicitComponent):
    """
    RAFT OpenMDAO Wrapper for platform configuration

    """
    def initialize(self):
        '''
        Inintialize turbine configurations as an 'om.ExplicitComponent.options'
        '''
        self.options.declare('member_options', default=design['platform'] , types=dict)

    def setup(self):

        members_opt = self.options['member_options']      

        #-------------------------------------input-----------------------------------------#
        
        # share_input
        self.add_input('turbine_tower_rA_z',   val=0.0, units='m', shape=1, desc='z coordinate of top end of center column, connect to turbine_tower_rA_z')
        
        # cener_column
        self.add_input('center_column_d',         val=0.0, units='m', shape=1, desc='z coordinate of rotor hub of each rotor') 
        # self.add_input('center_column_t',   val=0.0, units='m', shape=1, desc='z coordinate of tower base of each rotor')

        # outer_column
        # '''Temporally assume columns share the same freeboard'''
        self.add_input('outer_column_rB_z',   val=0.0, units='m', shape=1, desc='z coordinate of top end of each outer column')
        self.add_input('outer_column_d',      val=0.0, units='m', shape=1, desc='z coordinate of top end of center column, connect to turbine_tower_rA_z')
        # self.add_input('outer_column_t',      val=0.0, units='m', shape=1, desc='z coordinate of top end of center column, connect to turbine_tower_rA_z')
        
        # outer_column_up
        self.add_input('outer_column_up_offset',    val=0.0, units='m',   shape=1, desc='offset in [m] between center of column and ref point')
        self.add_input('outer_column_up_heading',   val=0.0, units='deg', shape=1, desc='offset in [deg] between center of column and ref point')
        self.add_input('outer_column_up_l_fill',    val=0.0, units='m',   shape=1, desc='fill level in [m]')
        
        # outer_column_down
        self.add_input('outer_column_down_offset',    val=0.0, units='m', shape=1, desc='offset in [m] between center of column and ref point')
        self.add_input('outer_column_down_l_fill',    val=0.0, units='m', shape=1, desc='fill level in [m]')
        
        # pontoon
        self.add_input('pontoon_sl_0',   val=0.0, units='m', shape=1, desc='z coordinate of top end of each outer column')
        '''Temporally assume pontoon_sl_1 = outer_column_d'''
        # self.add_input('pontoon_sl_1',   val=0.0, units='m', shape=1, desc='z coordinate of top end of each outer column')
        # self.add_input('pontoon_t',      val=0.0, units='m', shape=1, desc='z coordinate of top end of each outer column')

        # upper_support
        self.add_input('upper_support_d',   val=0.0, units='m', shape=1, desc='z coordinate of top end of each outer column')


        #-------------------------------------output-----------------------------------------#
        # inertia
        self.add_output("platform_displacement", 0.0, desc='Volumetric platform displacement',        units='m**3')
        self.add_output('mass',          val=0.0, shape=1, desc='mass of whole fowt without mooring', units='kg')
        self.add_output('platform_mass', val=0.0, shape=1, desc='mass of whole platform'            , units='kg' )
        self.add_output('rCG_sub',       val=np.zeros(3),  desc='rCG of platform'                   , units='m' )
        self.add_output('rCG',           val=np.zeros(3),  desc='rCG of FOWT'                       , units='m' )
        
        # static stability
        self.add_input('rCB',            val=np.zeros(3),  desc='rCB of FOWT'                       , units='m' )
        self.add_output('platform_GM',   val=0.0, shape=1, desc='metacentirc center of platform'    , units='m' )


        # constraint
        # self.add_output('con1', val=0.0, shape=1, desc='static stability constraint'    , units='m'  )
        
        # object
        self.add_output('xCG_offset', val=0.0, shape=1, desc='x offset of CG to prp'    , units='m')

    
    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        members_opt = self.options['member_options']      

        # main_column
        members_opt[0]['rB'] = [0,0,inputs['turbine_tower_rA_z'].item()]
        members_opt[0]['d' ] = inputs['center_column_d'].item()

        # outer_column_up
        angle1 = inputs['outer_column_up_heading']
        angle2 = 360-inputs['outer_column_up_heading']

        members_opt[1]['rA'] = [inputs['outer_column_up_offset'].item(), 0, draft]
        members_opt[1]['rB'] = [inputs['outer_column_up_offset'].item(), 0, inputs['outer_column_rB_z'].item()]
        members_opt[1]['heading'] = [angle1.item(), angle2.item()]
        members_opt[1]['d' ] = inputs['outer_column_d'].item()
        members_opt[1]['l_fill'] = inputs['outer_column_up_l_fill'].item()

        # outer_column_down
        members_opt[2]['rA'] = [inputs['outer_column_down_offset'].item(), 0, draft]
        members_opt[2]['rB'] = [inputs['outer_column_down_offset'].item(), 0, inputs['outer_column_rB_z'].item()]
        members_opt[2]['d' ] = inputs['outer_column_d'].item()
        members_opt[2]['l_fill'] = inputs['outer_column_down_l_fill'].item()

        # pontoon_up
        xA = inputs['center_column_d']*0.5
        xB = inputs['outer_column_up_offset'] - 0.5*inputs['outer_column_d']
        zAB  = draft+inputs['pontoon_sl_0']*0.5

        members_opt[3]['rA'] = [xA.item(), 0, zAB.item()]
        members_opt[3]['rB'] = [xB.item(), 0, zAB.item()]
        members_opt[3]['heading'] = [angle1.item(), angle2.item()]
        members_opt[3]['sl'][0]  = inputs['pontoon_sl_0'].item()
        members_opt[3]['sl'][1]  = inputs['outer_column_d'].item()

        # upper_support_up
        zAB  = inputs['outer_column_rB_z'] - inputs['upper_support_d']*0.5

        members_opt[5]['rA'] = [xA.item(), 0, zAB.item()]
        members_opt[5]['rB'] = [xB.item(), 0, zAB.item()]
        members_opt[5]['heading'] = [angle1.item(), angle2.item()]
        members_opt[5]['d' ] = inputs['upper_support_d'].item()
        
        # pontoon_down
        xB = inputs['outer_column_down_offset'] - 0.5*inputs['outer_column_d']
        
        members_opt[4]['rA'] = [xA.item(), 0, zAB.item()]
        members_opt[4]['rB'] = [xB.item(), 0, zAB.item()]
        members_opt[4]['sl'][0]  = inputs['pontoon_sl_0'].item()
        members_opt[4]['sl'][1]  = inputs['outer_column_d'].item()

        # upper_support_down
        zAB  = inputs['outer_column_rB_z'] - inputs['upper_support_d']*0.5

        members_opt[6]['rA'] = [xA.item(), 0, zAB.item()]
        members_opt[6]['rB'] = [xB.item(), 0, zAB.item()]
        members_opt[6]['d' ] = inputs['upper_support_d'].item()

        # initialize a FOWT instance for statics calculations
        fowt = FOWT(design, np.linspace(min_freq, max_freq, 81), None, depth=200, x_ref=0, y_ref=0, heading_adjust=0)
        fowt.setPosition([0,0,0,0,0,0])

        fowt.calcStatics()

        outputs['platform_displacement'] = fowt.V
        outputs['mass'] = fowt.m
        outputs['platform_mass'] = fowt.m_sub
        outputs['rCG_sub'] = fowt.rCG_sub
        outputs['rCG'] = fowt.rCG
        
        outputs['rCB'] = fowt.rCB
        outputs['platform_GM'] = fowt.rM[2] - fowt.rCG[2]

        outputs['xCG_offset'] = np.abs(fowt.rCG[0])

        
        return super().compute(inputs, outputs, discrete_inputs, discrete_outputs)

if __name__ == '__main__':

    # from raft.raft_rotor import Rotor
    # rotor_list:Iterable[Rotor] = []
    # rotor_list = [Rotor(design['turbine'], [1], i) for i in range(2)]

    # import matplotlib.pyplot as plt
    # from raft.raft_fowt import FOWT

    # fowt = FOWT(design, np.linspace(min_freq, max_freq, 81), None, depth=600, x_ref=0, y_ref=0, heading_adjust=0)
    
    
    prob = om.Problem()
    # prob.model.add_subsystem('twin_rotor_comp', 
    #                     Rotors(), 
    #                     promotes_inputs=['turbine_L_rotors', 'turbine_zRNA', 'turbine_tower_rA_z'],
    #                     promotes_outputs=['max_tower_stress'])
    
    prob.model.add_subsystem('twin_rotor_comp', 
                             Rotors())

    # prob.model.list_inputs(val=True, units=True, shape=True)

    prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'shgo'
    # prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['maxiter'] = 100

    # prob.driver = om.pyOptSparseDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['optimizer'] = 'NSGA2'

    # prob.driver = om.SimpleGADriver()
    # prob.driver.options['maxiter'] = 100
    # prob.driver.options['optimizer'] = 'SNOPT'

    # prob.model.set_input_defaults('turbine_L_rotors', 78.0)
    # prob.model.set_input_defaults('turbine_zRNA', 80.0)
    # prob.model.set_input_defaults('turbine_tower_rA_z', 10.0)

    prob.model.add_design_var('twin_rotor_comp.turbine_L_rotors', lower=(Rtip*2)*1.05, upper=(Rtip*2)*1.5)
    prob.model.add_design_var('twin_rotor_comp.turbine_zRNA', lower=Rtip+15, upper=Zhub)
    prob.model.add_design_var('twin_rotor_comp.turbine_tower_rA_z', lower=10, upper=15)

    

    # prob.model.add_subsystem('const', 
    #                          om.ExecComp('g = turbine_zRNA - turbine_L_rotor/2 - turbine_tower_rA_z', units='m'))


    prob.model.add_constraint('twin_rotor_comp.con1', lower=0)

    prob.model.add_objective('twin_rotor_comp.max_tower_stress')

    prob.setup()
    prob.set_val('twin_rotor_comp.turbine_L_rotors', val=150, units='m')
    prob.set_val('twin_rotor_comp.turbine_zRNA', val=80.0, units='m')
    prob.set_val('twin_rotor_comp.turbine_tower_rA_z', val=10.0, units='m')

    # prob.run_model()
    # print(prob['twin_rotor_comp.max_tower_stress'])

    
    prob.run_driver()

    prob.model.list_inputs(val=True, units=True, shape=True)
    prob.model.list_outputs(val=True, units=True, shape=True)

    print(prob['twin_rotor_comp.max_tower_stress'])
    print(prob['twin_rotor_comp.turbine_tower_d'])
    print(prob['twin_rotor_comp.turbine_tower_t'])
    # print(prob['twin_rotor_comp.turbine_tower_t'])

    a = prob.model.twin_rotor_comp.options['turbine_options']

    
    
    
    
    
    
    fowt = FOWT(design, np.linspace(min_freq, max_freq, 81), None, depth=200, x_ref=0, y_ref=0, heading_adjust=0)
    fowt.setPosition([0,0,0,0,0,0])
    fowt.calcStatics()

    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    fowt.plot(ax, plot_ms=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    max_range = max(np.ptp(xlim), np.ptp(ylim), np.ptp(zlim))

    ax.set_xlim([np.mean(xlim) - max_range / 2, np.mean(xlim) + max_range / 2])
    ax.set_ylim([np.mean(ylim) - max_range / 2, np.mean(ylim) + max_range / 2])
    ax.set_zlim([np.mean(zlim) - max_range / 2, np.mean(zlim) + max_range / 2])

    ax.set_box_aspect([1, 1, 1]) 
    plt.show()

    a = 1


    # 

    # prob.run_model()
    # print(prob['twin_rotor_comp.max_tower_stress'])
    # ax = plt.figure().add_subplot(projection='3d')
    # for i in range(2):
    #     rotor_list[i].setPosition()
    #     rotor_list[i].plot(ax)

    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # zlim = ax.get_zlim()

    # max_range = max(np.ptp(xlim), np.ptp(ylim), np.ptp(zlim))

    # ax.set_xlim([np.mean(xlim) - max_range / 2, np.mean(xlim) + max_range / 2])
    # ax.set_ylim([np.mean(ylim) - max_range / 2, np.mean(ylim) + max_range / 2])
    # ax.set_zlim([np.mean(zlim) - max_range / 2, np.mean(zlim) + max_range / 2])
    # plt.show()


    # plt.plot(stations, d)
    # plt.show()

    a = 1