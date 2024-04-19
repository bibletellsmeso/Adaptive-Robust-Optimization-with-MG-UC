import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

from utils import read_file
from SP_primal_LP import *
from Params import PARAMETERS
from root_project import ROOT_DIR
from Data_read import *

class CCG_SP():
    """
    CCGD = Column and Constraint Gneration Dual
    SP = Sub Problem of the CCG dual cutting plane algorithm.
    SP = Max-min problem that is reformulated as a single max problem by taking the dual.
    The resulting maximization problem is bilinear and is linearized using big-M's values.
    The final reformulated SP is a MILP due to binary variables related to the uncertainty set.
    The integer variable(y_b) related with charge and discharge was relaxation.
    
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar PV_forecast: PV forecasts (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)

    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array, PV_pos:np.array, PV_neg:np.array, load_pos:np.array, load_neg:np.array, x_binary:np.array, x_startup:np.array, x_shutdown:np.array, gamma_PV:float=0, gamma_load:float=0, M_al_neg:float=None, M_al_pos:float=None, M_be_neg:float=None, M_be_pos:float=None):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        self.PV_forecast = PV_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        self.x_b = x_binary # (Binary)
        self.x_su = x_startup # (Binary)
        self.x_sd = x_shutdown # (Binary)
        self.PV_pos = PV_pos # (kW) The maximal deviation betwwen the min and forecast PV uncertainty set bounds
        self.PV_neg = PV_neg # (kW) The maximal deviation between the max and forecast PV uncertainty set bounds
        self.load_pos = load_pos # (kw) The maximal deviation between the min and forecast load uncertainty set bounds
        self.load_neg = load_neg # (kW) The maximal deviation between the max and forecast load uncertainty set bounds
        self.gamma_PV = gamma_PV # uncertainty budget <= self.nb_periods, gamma = 0: no uncertainty
        self.gamma_load = gamma_load
        self.M_al_pos = M_al_pos
        self.M_al_neg = M_al_neg # big-M value
        self.M_be_pos = M_be_pos
        self.M_be_neg = M_be_neg

        # Diesel parameters
        self.diesel_min = PARAMETERS['Diesel']['diesel_min'] # (kW)
        self.diesel_max = PARAMETERS['Diesel']['diesel_max'] # (kW)
        self.diesel_ramp_up = PARAMETERS['Diesel']['ramp_up'] #  (kW)
        self.diesel_ramp_down = PARAMETERS['Diesel']['ramp_down'] #  (kW)
        self.p_rate = PARAMETERS['Diesel']['p_rate']

        # BESS parameters
        self.BESScapacity = PARAMETERS['BESS']['BESS_capacity'] # (kWh)
        self.soc_ini = PARAMETERS['BESS']['soc_ini'] # (kWh)
        self.soc_end = PARAMETERS['BESS']['soc_end'] # (kWh)
        self.soc_min = PARAMETERS['BESS']['soc_min'] # (kWh)
        self.soc_max = PARAMETERS['BESS']['soc_max'] # (kWh)
        self.charge_eff = PARAMETERS['BESS']['charge_eff'] # (/)
        self.discharge_eff = PARAMETERS['BESS']['discharge_eff'] # (/)
        self.charge_power = PARAMETERS['BESS']['charge_power'] # (kW)
        self.discharge_power = PARAMETERS['BESS']['discharge_power'] # (kW)

        # Cost parameters
        self.cost_su = PARAMETERS['cost']['cost_start_up']
        self.cost_sd = PARAMETERS['cost']['cost_shut_down']
        self.cost_fuel = PARAMETERS['cost']['cost_of_fuel']
        self.cost_a = PARAMETERS['cost']['a_of_dg']
        self.cost_b = PARAMETERS['cost']['b_of_dg']
        self.OM_dg = PARAMETERS['cost']['O&M_of_dg']
        self.OM_BESS = PARAMETERS['cost']['O&M_of_BESS']
        self.OM_PV = PARAMETERS['cost']['O&M_of_PV']
        self.cost_penalty = PARAMETERS['cost']['penalty_of_PV']

        self.curt = data.PV_pred

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Sovle model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("SP_dual_MILP")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create dual variables -> phi

        # 2.1 Continuous variables
        # primal constraints <= b -> dual variables <= 0, primal constraints = b -> dual varialbes are free, (primal constraints >= b -> dual variables >= 0)
        phi_DGmin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=-0, name="phi_DGmin")
        phi_DGmax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_DGmax")
        phi_DGdown = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_DGdown") # num: 95
        phi_DGup = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_DGup") # num: 95
        phi_bal = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_bal") # free dual of power balance
        phi_charge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_charge")
        phi_discharge = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_discharge")
        phi_ini = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_ini") # free of dual variable 
        phi_s = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_s") # num: 95, free dual of BESS dynamics (=)
        phi_end = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_end") # free of dual variable 
        phi_Smin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smin")
        phi_Smax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smax")
        phi_curt = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_curt")
        phi_PV = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_PV") # free of dual variable
        phi_load = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_load") # free of dual variable
  
        # 2.2 Continuous variables related to the uncertainty set
        epsilon_pos = model.addVars(self.nb_periods, vtype=GRB.BINARY, obj=0, name="epsilon_pos")
        epsilon_neg = model.addVars(self.nb_periods, vtype=GRB.BINARY, obj=0, name="epsilon_neg")
        delta_pos = model.addVars(self.nb_periods, vtype=GRB.BINARY, obj=0, name="delta_pos")
        delta_neg = model.addVars(self.nb_periods, vtype=GRB.BINARY, obj=0, name="delta_neg")

        # 2.3 Continuous varialbes use for the linearization of the bilinear terms
        alpha_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_pos')
        alpha_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_neg')
        beta_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_pos')
        beta_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_neg')

        # -------------------------------------------------------------------------------------------------------------

        # 3. create objective
        obj_exp = 0
        for i in range(self.nb_periods - 1):
            obj_exp += phi_DGdown[i] * self.diesel_ramp_down * self.period_hours + phi_DGup[i] * self.diesel_ramp_up * self.period_hours
        for i in self.t_set:
            obj_exp += - phi_DGmin[i] * self.x_b[i] * self.diesel_min + phi_DGmax[i] * self.x_b[i] * self.diesel_max
            obj_exp += phi_charge[i] * self.charge_power + phi_discharge[i] * self.discharge_power - self.soc_min * phi_Smin[i] + self.soc_max * phi_Smax[i]
            obj_exp += phi_PV[i] * self.PV_forecast[i] + phi_load[i] * self.load_forecast[i] + phi_curt[i] * self.curt[i]
            obj_exp += alpha_pos[i] * self.PV_pos[i] + alpha_neg[i] * self.PV_neg[i] + beta_pos[i] * self.load_pos[i] + beta_neg[i] * self.load_neg[i] # uncertainty set
            obj_exp += self.period_hours * self.cost_fuel * self.cost_b * self.p_rate * self.x_b[i] # a constant offset
            obj_exp += self.cost_su * self.x_su[i] + self.cost_sd * self.x_sd[i] # a constant offset
        obj_exp += phi_ini * self.soc_ini + phi_end * self.soc_end

        model.setObjective(obj_exp, GRB.MAXIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # primal variables >= 0 -> dual constraints <= c, primal variables are free -> dual constraints = c, (primal variables <= 0 -> dual constraints >= c)
        # Constraints related to the Diesel
        model.addConstr((phi_bal[0] - phi_DGmin[0] + phi_DGmax[0] + phi_DGdown[0] - phi_DGup[0] <= self.period_hours * (self.cost_fuel * self.cost_a + self.OM_dg)), name='c_diesel_first')
        model.addConstrs((phi_bal[i] - phi_DGmin[i] + phi_DGmax[i] - phi_DGdown[i - 1] + phi_DGdown[i] + phi_DGup[i - 1] - phi_DGup[i] <= self.period_hours * (self.cost_fuel * self.cost_a + self.OM_dg) for i in range(1, self.nb_periods - 1)), name='c_diesel')
        model.addConstr((phi_bal[self.nb_periods - 1] - phi_DGmin[self.nb_periods - 1] + phi_DGmax[self.nb_periods - 1] - phi_DGdown[self.nb_periods - 2] + phi_DGup[self.nb_periods - 2] <= self.period_hours * (self.cost_fuel * self.cost_a + self.OM_dg)), name='c_diesel_last')

        # Constraints related to the BESS
        model.addConstr(phi_charge[0] - phi_bal[0] <= self.period_hours * self.OM_BESS, name='c_charge_first') # time period 1
        model.addConstrs((phi_charge[i] - phi_bal[i] - phi_s[i - 1] * self.period_hours * self.charge_eff <= self.period_hours * self.OM_BESS for i in range(1, self.nb_periods)), name='c_charge') # time period 2 to nb_periods
        model.addConstr(phi_discharge[0] + phi_bal[0] <= self.period_hours * self.OM_BESS, name='c_discharge_first') # time period 1
        model.addConstrs((phi_discharge[i] + phi_bal[i] + phi_s[i - 1] * self.period_hours / self.discharge_eff <= self.period_hours * self.OM_BESS for i in range(1, self.nb_periods)), name='c_discharge') # time period 2 to nb_periods

        # Constraints related to the Dynamics of BESSx
        model.addConstr(- phi_Smin[0] + phi_Smax[0] + phi_ini - phi_s[0] <= 0, name='c_s_first') # time period 1 for phi_Smin/phi_Smax and time period 2 for phi_s
        model.addConstrs((- phi_Smin[i] + phi_Smax[i] + phi_s[i - 1] - phi_s[i] <= 0 for i in range(1, self.nb_periods - 1)), name='c_s') # time period 3 to nb_periods - 1
        model.addConstr(- phi_Smin[self.nb_periods - 1] + phi_Smax[self.nb_periods - 1] + phi_end + phi_s[self.nb_periods - 2] <= 0, name='c_s_last') # Last time period

        # Constraints related to PV and load
        model.addConstrs((phi_PV[i] + phi_bal[i] <= self.period_hours * self.OM_PV for i in self.t_set), name='c_PV')
        model.addConstrs((phi_load[i] - phi_bal[i] <= 0 for i in self.t_set), name='c_load')
        model.addConstrs((phi_curt[i] - phi_bal[i] <= self.period_hours * self.cost_penalty for i in self.t_set), name='c_curt')

        # Constraints related to the uncertainty set
        model.addConstrs((epsilon_pos[i] * epsilon_neg[i] == 0 for i in self.t_set), name='c_epsilon')
        model.addConstrs((delta_pos[i] * delta_neg[i] == 0 for i in self.t_set), name='c_delta')

        # Constraints related to the uncertainty budget
        model.addConstr(gp.quicksum(epsilon_pos[i] + epsilon_neg[i] for i in self.t_set) <= self.gamma_PV, name='c_PV_gamma') # PV uncertainty budget
        model.addConstr(gp.quicksum(delta_pos[i] + delta_neg[i] for i in self.t_set) <= self.gamma_load, name='c_demmand_gamma') # load uncertainty budget

        # Constraints required to linearize bilinear terms
        # 1. PV: alpha_pos = epsilon_pos * phi_PV, 
        model.addConstrs((alpha_pos[i] >= - self.M_al_pos * epsilon_pos[i] for i in self.t_set), name='c_alpha_pos_1_min')
        model.addConstrs((alpha_pos[i] <= self.M_al_pos * epsilon_pos[i] for i in self.t_set), name='c_alpha_pos_1_max')
        model.addConstrs((alpha_pos[i] >= phi_PV[i] - self.M_al_pos * (1 - epsilon_pos[i]) for i in self.t_set), name='c_alpha_pos_2_min')
        model.addConstrs((alpha_pos[i] <= phi_PV[i] + self.M_al_pos * (1 - epsilon_pos[i]) for i in self.t_set), name='c_alpha_pos_2_max')

        # 2. PV: alpha_neg = - epsilon_neg * phi_PV
        model.addConstrs((alpha_neg[i] >= - self.M_al_neg * epsilon_neg[i] for i in self.t_set), name='c_alpha_neg_1_min')
        model.addConstrs((alpha_neg[i] <= self.M_al_neg * epsilon_neg[i] for i in self.t_set), name='c_alpha_neg_1_max')        
        model.addConstrs((alpha_neg[i] >= - phi_PV[i] - self.M_al_neg * (1 - epsilon_neg[i]) for i in self.t_set), name='c_alpha_neg_2_min')
        model.addConstrs((alpha_neg[i] <= - phi_PV[i] + self.M_al_neg * (1 - epsilon_neg[i]) for i in self.t_set), name='c_alpha_neg_2_max')

        # 3. load: beta_pos = delta_pos * phi_de
        model.addConstrs((beta_pos[i] >= - self.M_be_pos * delta_pos[i] for i in self.t_set), name='c_beta_pos_1_min')
        model.addConstrs((beta_pos[i] <= self.M_be_pos * delta_pos[i] for i in self.t_set), name='c_beta_pos_1_max')        
        model.addConstrs((beta_pos[i] >= phi_load[i] - self.M_be_pos * (1 - delta_pos[i]) for i in self.t_set), name='c_beta_pos_2_min')
        model.addConstrs((beta_pos[i] <= phi_load[i] + self.M_be_pos * (1 - delta_pos[i]) for i in self.t_set), name='c_beta_pos_2_max')

        # 4. beta_neg = - delta_neg * phi_de
        model.addConstrs((beta_neg[i] >= - self.M_be_neg * delta_neg[i] for i in self.t_set), name='c_beta_neg_1_min')
        model.addConstrs((beta_neg[i] <= self.M_be_neg * delta_neg[i] for i in self.t_set), name='c_beta_neg_1_max')        
        model.addConstrs((beta_neg[i] >= - phi_load[i] - self.M_be_neg * (1 - delta_neg[i]) for i in self.t_set), name='c_beta_neg_2_min')
        model.addConstrs((beta_neg[i] <= - phi_load[i] + self.M_be_neg * (1 - delta_neg[i]) for i in self.t_set), name='c_beta_neg_2_max')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['phi_DGmin'] = phi_DGmin
        self.allvar['phi_DGmax'] = phi_DGmax
        self.allvar['phi_DGdown'] = phi_DGdown
        self.allvar['phi_DGup'] = phi_DGup
        self.allvar['phi_bal'] = phi_bal
        self.allvar['phi_charge'] = phi_charge
        self.allvar['phi_discharge'] = phi_discharge
        self.allvar['phi_s'] = phi_s
        self.allvar['phi_Smin'] = phi_Smin
        self.allvar['phi_Smax'] = phi_Smax
        self.allvar['phi_ini'] = phi_ini
        self.allvar['phi_end'] = phi_end
        self.allvar['phi_curt'] = phi_curt
        self.allvar['phi_PV'] = phi_PV
        self.allvar['phi_load'] = phi_load
        self.allvar['epsilon_pos'] = epsilon_pos
        self.allvar['epsilon_neg'] = epsilon_neg
        self.allvar['delta_pos'] = delta_pos
        self.allvar['delta_neg'] = delta_neg
        self.allvar['alpha_pos'] = alpha_pos
        self.allvar['alpha_neg'] = alpha_neg
        self.allvar['beta_pos'] = beta_pos
        self.allvar['beta_neg'] = beta_neg

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program %gs" % self.time_building_model)
        
        return model
    
    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):
        """
        :param LogToConsole: no log in the console if set to False.
        :param logfile: no log in file if set to ""
        :param Threads: Default value = 0 -> use all threads
        :param MIPFocus: If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
                        If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
                        If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.
        :param TimeLimit: in seconds.
        """

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        self.model.setParam('LogFile', logfile)
        self.model.setParam('Threads', Threads)

        self.model.optimize()

        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 0 dimensional variables
        for var in ['phi_ini', 'phi_end']:
            solution[var] = self.allvar[var].X

        # 1 dimensional variables
        for var in ['phi_DGmin', 'phi_DGmax', 'phi_bal', 'phi_charge', 'phi_discharge', 'phi_Smin', 'phi_Smax', 
                    'phi_curt', 'phi_PV', 'phi_load', 'epsilon_pos', 'epsilon_neg', 'delta_pos', 'delta_neg', 'alpha_pos', 'alpha_neg', 'beta_pos', 'beta_neg']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]

        for var in ['phi_DGdown', 'phi_DGup', 'phi_s']:
            solution[var] = [self.allvar[var][t].X for t in range(self.nb_periods - 1)]

        # 6. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution
    
    def export_model(self, filename):
        """
        Export the model into a lp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/'
    day = '2018-07-04'

    PV_solution = np.array(pd.read_csv('PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
    load_solution = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
    PV_forecast = data.PV_pred
    load_forecast = data.load_egg
    
    x_binary = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/', name='sol_LP_x_b')
    x_startup = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/', name='sol_LP_x_su')
    x_shutdown = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/', name='sol_LP_x_sd')

    PV_min = PV_forecast - data.PV_neg
    PV_max = PV_forecast + data.PV_pos
    load_min = load_forecast + data.load_neg
    load_max = load_forecast - data.load_pos
    PV_neg = data.PV_neg
    PV_pos = data.PV_pos
    load_neg = data.load_neg
    load_pos = data.load_pos

    gamma_PV = 96
    gamma_load = 96
    M_al_neg = M_al_pos = M_be_neg = M_be_pos = 1

    SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_pos=PV_pos, PV_neg=PV_neg, load_pos=load_pos, load_neg=load_neg, x_binary=x_binary, x_startup=x_startup, x_shutdown=x_shutdown, gamma_PV=gamma_PV, gamma_load=gamma_load, M_al_neg=M_al_neg, M_al_pos=M_al_pos, M_be_neg=M_be_neg, M_be_pos=M_be_pos)
    SP_dual.export_model(dirname + 'SP_dual_MILP')
    MIPFocus = 0
    TimeLimit = 10
    logname = 'SP_dual_MILP_start_' + 'MIPFocus_' + str(MIPFocus) + '.log'
    SP_dual.solve(LogToConsole=True, logfile=dirname + logname, Threads=1, MIPFocus=MIPFocus, TimeLimit=TimeLimit)
    solution = SP_dual.store_solution()

    print('nominal objective %.2f' % (solution['obj']))
    plt.style.use(['science', 'no-latex'])
    
    plt.figure()
    plt.plot(solution['phi_PV'], label='phi_PV')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['phi_load'], label='phi_load')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['epsilon_pos'], label='epsilon_pos')
    plt.plot(solution['epsilon_neg'], label='epsilon_neg')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['delta_pos'], label='delta_pos')
    plt.plot(solution['delta_neg'], label='delta_neg')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['alpha_pos'], label='alpha_pos')
    plt.plot(solution['alpha_neg'], label='alpha_neg')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['beta_pos'], label='beta_pos')
    plt.plot(solution['beta_neg'], label='beta_neg')
    plt.legend()
    plt.show()

    PV_worst_case = [PV_forecast[i] + PV_pos[i] * solution['epsilon_pos'][i] - PV_neg[i] * solution['epsilon_neg'][i] for i in range(96)]
    load_worst_case = [load_forecast[i] + load_pos[i] * solution['delta_pos'][i] - load_neg[i] * solution['delta_neg'][i] for i in range(96)]

    plt.figure()
    plt.plot(PV_worst_case, marker='.', color='k', label='PV_worst_case')
    # plt.plot(PV_solution, label = 'Pm')
    plt.plot(PV_forecast, label = 'Pp')
    plt.plot(PV_forecast - PV_neg, ':', label = 'PV min')
    plt.plot(PV_forecast + PV_pos, ':', label = 'PV max')
    plt.ylim(-0.05 * PARAMETERS['PV_capacity'], PARAMETERS['PV_capacity'])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(load_worst_case, marker='.', color='k', label='load_worst_case')
    plt.plot(load_solution, label = 'Dm')
    plt.plot(load_forecast, label = 'Dp')
    plt.plot(load_forecast - load_neg, ':', label = 'PV min')
    plt.plot(load_forecast + load_pos, ':', label = 'PV max')
    # plt.ylim(-0.05 * PARAMETERS['PV_capacity'], PARAMETERS['PV_capacity'])
    plt.legend()
    plt.show()

    # Get dispatch variables by solving the primal LP with the worst case PV, load generation trajectory
    SP_primal = SP_primal_LP(PV_forecast=PV_worst_case, load_forecast=load_worst_case, x_binary=x_binary, x_startup=x_startup, x_shutdown=x_shutdown)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    print('RO objective %.2f' % (SP_primal_sol['obj']))

    plt.figure()
    plt.plot(SP_primal_sol['y_s'], label='soc')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.legend()
    plt.show()

