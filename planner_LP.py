import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from Params import PARAMETERS
from utils import dump_file
from utils import read_file
from Data_read import *


class Planner_LP():
    """
    LP capacity firming formulation: no binary variables to ensure not simultaneous charge and discharge.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar PV_forecast: PV point forecasts (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array, x_binary:np.array=None):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours'] # 1/4 hour
        self.nb_periods = int(24 / self.period_hours) # 96
        self.t_set = range(self.nb_periods)

        # Parameters required for the MP in the CCG algorithm
        self.PV_forecast = PV_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        self.x_binary = x_binary # (on/off)

        # Diesel parameters
        self.diesel_min = PARAMETERS['Diesel']['diesel_min'] # (kW)
        self.diesel_max = PARAMETERS['Diesel']['diesel_max'] # (kW)
        self.diesel_ramp_up = PARAMETERS['Diesel']['ramp_up'] #  (kW)
        self.diesel_ramp_down = PARAMETERS['Diesel']['ramp_down'] #  (kW)
        self.p_rate = PARAMETERS['Diesel']['p_rate']

        # BESS parameters
        self.BESScapacity = PARAMETERS['BESS']['BESS_capacity']  # (kWh)
        self.soc_ini = PARAMETERS['BESS']['soc_ini']  # (kWh)
        self.soc_end = PARAMETERS['BESS']['soc_end']  # (kWh)
        self.soc_min = PARAMETERS['BESS']['soc_min']  # (kWh)
        self.soc_max = PARAMETERS['BESS']['soc_max']  # (kWh)
        self.charge_eff = PARAMETERS['BESS']['charge_eff']  # (/)
        self.discharge_eff = PARAMETERS['BESS']['discharge_eff']  # (/)
        self.charge_power = PARAMETERS['BESS']['charge_power']  # (kW)
        self.discharge_power = PARAMETERS['BESS']['discharge_power']  # (kW)

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

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. Create model
        model = gp.Model("planner_MILP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create vairables
        # 2.1 Create First-stage variables -> x
        x_su = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_su') # start-up
        x_sd = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_sd') # shut-down
        x_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_b') # 1 -> on, 0 -> off
        
        # Set the x_binary variable to the values in self.x_b
        if self.x_binary is not None:
            for i in self.t_set:
                x_b[i].setAttr("ub", self.x_binary[i])
                x_b[i].setAttr("lb", self.x_binary[i])

        # 2.2 Second-stage variables -> y
        # Diesel power (kW)
        y_diesel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_diesel")
        # State of charge of the battery (kWh)
        y_s = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s")
        # Charging power (kW)
        y_charge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge")
        # Discharging power (kW)
        y_discharge = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge")
        # binary variable -> y_b = 1: charge / y_b = 0: discharge
        # y_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b_")
        # PV generation (kW)
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        # PV curtailment (kW)
        y_curt = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt")
        # Load load (kW)
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")
        # PV piecewise curtailment (kW)
        # y_curt_pie = model.addVars(self.N_PWL, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_pie")

        # -------------------------------------------------------------------------------------------------------------
        # diesel start_up cost
        su_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='diesel_su_cost')
        # diesel shut_down cost
        sd_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='diesel_sd_cost')
        # diesel fuel cost
        fuel_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="diesel_cost")
        # diesel O&M cost
        OM_dg_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_diesel_cost")
        # BESS O&M cost
        OM_BESS_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_BESS_cost")
        # PV O&M cost
        OM_PV_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_PV_cost")
        # PV penalty cost
        penalty_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="PV_penalty_cost")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective
        obj = gp.quicksum(su_cost[i] + sd_cost[i] + fuel_cost[i] + OM_dg_cost[i] + OM_BESS_cost[i] + OM_PV_cost[i]  + penalty_cost[i] for i in self.t_set)
        model.setObjective(obj, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # 4.1 Fisrt stage constraints
        # Diesel strat-up, shut-down cst
        model.addConstrs((x_su[i] + x_sd[i] <= 1 for i in self.t_set), name='c_diesel_su-sd')
        # Diesel on/off cst
        model.addConstrs((x_su[i] - x_sd[i] == x_b[i] - x_b[i-1] for i in range(1, self.nb_periods)), name='c_x_b')

        # 4.2 Second stage constraints
        # 4.2.1 Diesel
        # min diesel cst
        model.addConstrs((y_diesel[i] >= self.diesel_min * x_b[i] for i in self.t_set), name='c_min_diesel')
        # max diesel cst
        model.addConstrs((y_diesel[i] <= self.diesel_max * x_b[i] for i in self.t_set), name='c_max_diesel')
        # diesel ramp down cst
        model.addConstrs((- y_diesel[i] + y_diesel[i-1] <= self.diesel_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-down')
        # diesel ramp up cst
        model.addConstrs((y_diesel[i] - y_diesel[i-1] <= self.diesel_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-up')

        # 4.2.2 cost cst
        # diesel start_up cost
        model.addConstrs((su_cost[i] == self.cost_su * x_su[i] for i in self.t_set), name='c_diesel_start_up')
        # diesel shut_down cost
        model.addConstrs((sd_cost[i] == self.cost_sd * x_sd[i] for i in self.t_set), name='c_diesel_shut_down')
        # diesel fuel cost
        model.addConstrs((fuel_cost[i] == self.period_hours * (self.cost_fuel * ((self.cost_a * y_diesel[i]) + self.cost_b * self.p_rate * x_b[i])) for i in self.t_set), name='c_diesel_fuel')
        # diesel O&M cost
        model.addConstrs((OM_dg_cost[i] == self.period_hours * self.OM_dg * y_diesel[i] for i in self.t_set), name='c_diesel_O&M')
        # BESS O&M cost
        model.addConstrs((OM_BESS_cost[i] == self.period_hours * self.OM_BESS * (y_charge[i] + y_discharge[i]) for i in self.t_set), name='c_BESS_O&M')
        # PV O&M cost
        model.addConstrs((OM_PV_cost[i] == self.period_hours * self.OM_PV * y_PV[i] for i in self.t_set), name='c_PV_O&M')
        # PV penalty cost
        model.addConstrs((penalty_cost[i] == self.period_hours * self.cost_penalty * y_curt[i] for i in self.t_set), name='c_PV_penalty')

        # 4.2.2 power balance equation
        model.addConstrs((y_diesel[i] + y_PV[i] - y_curt[i] + (y_discharge[i] - y_charge[i]) - y_load[i] == 0 for i in self.t_set), name='c_power_balance_eq')
        
        # 4.2.3 BESS
        # max charge cst
        model.addConstrs((y_charge[i] <= self.charge_power for i in self.t_set), name='c_max_charge')
        # max discharge cst
        model.addConstrs((y_discharge[i] <= self.discharge_power for i in self.t_set), name='c_max_discharge')
        # min soc cst
        model.addConstrs((y_s[i] >= self.soc_min for i in self.t_set), name='c_min_s')
        # min soc cst
        model.addConstrs((y_s[i] <= self.soc_max for i in self.t_set), name='c_max_s')
        # BESS dynamics first period
        model.addConstr((y_s[0] == self.soc_ini), name='c_BESS_first_period')
        # BESS dynamics from second to last periods
        model.addConstrs((y_s[i] - y_s[i - 1] - self.period_hours * ((self.charge_eff * y_charge[i]) - (y_discharge[i] / self.discharge_eff)) == 0
                          for i in range(1, self.nb_periods)), name='c_BESS_dynamics')
        # BESS dynamics last period
        model.addConstr((y_s[self.nb_periods - 1] == self.soc_end), name='c_BESS_last_period')

        # 4.2.4 PV curtailment
        model.addConstrs((y_curt[i] <= self.curt[i] for i in self.t_set), name='c_curtailment')
        # # PV curtailment 10 piece
        '''
        model.addConstr((y_curt_pie[i] <= 400/10 for i in range(10)), name='c_max_curt_piece')
        model.addConstr((y_curt_pie[i] >= 0 for i in range(10)), name='c_min_curt_piece')
        '''

        # 4.2.5 PV generation cst
        model.addConstrs((y_PV[i] == PV_forecast[i] for i in self.t_set), name='c_PV_generation')
        # 4.2.6 load cst
        model.addConstrs((y_load[i] == self.load_forecast[i] for i in self.t_set), name='c_load')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['x_su'] = x_su
        self.allvar['x_sd'] = x_sd
        self.allvar['x_b'] = x_b
        self.allvar['y_diesel'] = y_diesel
        self.allvar['su_cost'] = su_cost
        self.allvar['sd_cost'] = sd_cost
        self.allvar['fuel_cost'] = fuel_cost
        self.allvar['OM_dg_cost'] = OM_dg_cost
        self.allvar['OM_BESS_cost'] = OM_BESS_cost
        self.allvar['OM_PV_cost'] = OM_PV_cost
        self.allvar['penalty_cost'] = penalty_cost
        self.allvar['y_s'] = y_s
        self.allvar['y_charge'] = y_charge
        self.allvar['y_discharge'] = y_discharge
        # self.allvar['y_b'] = y_b
        self.allvar['y_PV'] = y_PV
        self.allvar['y_curt'] = y_curt
        self.allvar['y_load'] = y_load
        '''self.allvar['y_curt_pie'] = y_curt_pie'''

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def solve(self, outputflag:bool=False):
        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.solver_stats = self.model.status
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):
        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 1 dimensional variables
        for var in ['x_su', 'x_sd', 'x_b', 'y_diesel', 'su_cost', 'sd_cost', 'fuel_cost', 'y_s', 'OM_dg_cost', 'OM_BESS_cost', 'OM_PV_cost',
                    'penalty_cost', 'y_charge', 'y_discharge', 'y_PV', 'y_curt', 'y_load']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]

        # Timing indicators
        solution['time_building'] = self.time_building_model
        solution['time_solving'] = self.time_solving_model
        solution['time_total'] = self.time_building_model + self.time_solving_model

        num_violated_constraints = m.ConstrVio
        print("Number of violated constraints: ", num_violated_constraints)

        return solution
    
    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG'
    day = '2018-07-04'

    PV_solution = np.array(pd.read_csv('PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
    load_solution = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
    PV_forecast = data.PV_pred
    load_forecast = data.load_egg

    planner_perfect = Planner_LP(PV_forecast=PV_solution, load_forecast=load_solution)
    planner_perfect.export_model(dirname + 'LP')
    planner_perfect.solve()
    solution_perfect = planner_perfect.store_solution()

    print('objective oracle %.2f' % (solution_perfect['obj']))

    dump_file(dir=dirname, name='sol_LP_oracle', file=solution_perfect['x_b'])

    plt.figure()
    plt.plot(solution_perfect['y_charge'], label='y_charge')
    plt.plot(solution_perfect['y_discharge'], label='y_discharge')
    plt.plot(solution_perfect['y_s'], label='y s')
    plt.legend()
    plt.show()
    
    planner = Planner_LP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    planner.solve()
    solution = planner.store_solution()
    dump_file(dir=dirname, name='sol_LP_x_b', file=solution['x_b'])
    dump_file(dir=dirname, name='sol_LP_x_su', file=solution['x_su'])
    dump_file(dir=dirname, name='sol_LP_x_sd', file=solution['x_sd'])

    plt.figure()
    plt.plot(solution['x_b'], label='x fist-stage')
    plt.plot(solution['y_diesel'], label='diesel')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_discharge'], solution['y_charge'])]), label='BESS')
    plt.plot(solution['y_load'], label='load')
    plt.plot(solution['y_PV'], label='PV generation')
    plt.plot(solution['y_curt'], label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_PV'], solution['y_curt'])]), label='PV output')
    plt.title('LP formulation')
    plt.legend()
    # plt.savefig(dirname+ 'LP_oracle_vs_PVUSA.pdf')
    plt.show()


