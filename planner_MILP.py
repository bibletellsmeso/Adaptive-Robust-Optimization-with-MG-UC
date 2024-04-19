import math
import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from utils import dump_file
from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from Data_read import *
from Params import PARAMETERS

class Planner_MILP():
    """
    MILP capacity firming formulation: binary variables to avoid simultaneous charge and discharge.
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
        self.diesel_ramp_up = PARAMETERS['Diesel']['ramp_up'] # (kW)
        self.diesel_ramp_down = PARAMETERS['Diesel']['ramp_down'] # (kW)
        self.p_rate = PARAMETERS['Diesel']['p_rate']

        # BESS parameters
        self.BESScapacity = PARAMETERS['BESS']['BESS_capacity']  # (kWh)
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
        x_su = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="x_su") # start-up
        x_sd = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="x_sd") # shut-down
        x_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="x_b") # 1 -> on, 0 -> off
        
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
        y_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b_")
        # PV generation (kW)
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        # PV curtailment (kW)
        y_curt = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt")
        # Load load (kW)
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")
        # PV piecewise curtailment (kW)
        # y_curt_pie = model.addVars(self.N_PWL, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_pie")

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
        obj = gp.quicksum(su_cost[i] + sd_cost[i] + fuel_cost[i] + OM_dg_cost[i] + OM_BESS_cost[i] + OM_PV_cost[i] + penalty_cost[i] for i in self.t_set)
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
        # model.addConstrs(fuel_cost[i] >= (self.fuel_cost[p + 1] - self.fuel_cost[p]) / (self.diesel_power[p + 1] - self.diesel_power[p]) * y_diesel[i] for i in self.t_set for p in range(10))
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

        # 4.2.3 power balance equation
        model.addConstrs((y_diesel[i] + y_PV[i] - y_curt[i] + y_discharge[i] - y_charge[i] - y_load[i] == 0 for i in self.t_set), name='c_power_balance_eq')
        
        # 4.2.4 BESS
        # max charge cst
        model.addConstrs((y_charge[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_max_charge')
        # max discharge cst
        model.addConstrs((y_discharge[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set), name='c_max_discharge')
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
        # 4.2.5 PV curtailment
        model.addConstrs((y_curt[i] <= self.curt[i] for i in self.t_set), name='c_curtailment')
        # # PV curtailment 10 piece
        '''
        model.addConstr((y_curt_pie[i] <= 400/10 for i in range(10)), name='c_max_curt_piece')
        model.addConstr((y_curt_pie[i] >= 0 for i in range(10)), name='c_min_curt_piece')
        '''
        # 4.2.6 PV generation cst
        model.addConstrs((y_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_generation')
        # 4.2.7 load cst
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
        self.allvar['y_b'] = y_b
        self.allvar['y_PV'] = y_PV
        self.allvar['y_curt'] = y_curt
        self.allvar['y_load'] = y_load
        '''self.allvar['y_curt_pie'] = y_curt_pie'''

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):

        t_solve = time.time()

        self.model.setParam('LogToConsole', LogToConsole) # no log in the console if set to False
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        # self.model.setParam('DualReductions', 0) # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.

        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        self.model.setParam('LogFile', logfile) # no log in file if set to ""
        self.model.setParam('Threads', Threads) # Default value = 0 -> use all threads

        self.model.optimize()
        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            solution['obj'] = m.objVal

            # 1 dimensional variables
            for var in ['x_su', 'x_sd', 'x_b', 'y_diesel', 'su_cost', 'sd_cost', 'fuel_cost', 'y_s', 'OM_dg_cost', 'OM_BESS_cost', 'OM_PV_cost',
                        'penalty_cost', 'y_charge', 'y_discharge', 'y_PV', 'y_curt', 'y_load', 'y_b']:
                solution[var] = [self.allvar[var][t].X for t in self.t_set]
        else:
            print('WARNING planner MILP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            solution['obj'] = math.nan

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)


# Validation set
VS = 'VS1' # 'VS1', 'VS2

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/export_MILP/'

    # Compute the PV, load min and max 
    PV_min = data.PV_min
    PV_max = data.PV_max
    load_min = data.load_max
    load_max = data.load_min

    # load data
    PV_forecast = data.PV_pred
    load_forecast = data.load_egg
    PV_oracle = data.PV_oracle
    load_oracle = data.load_oracle
    PV_worst = data.PV_worst
    load_worst = data.load_worst

    day = '2018-07-04'

    # Plot point forecasts vs observations
    FONTSIZE = 20
    plt.style.use(['science', 'no-latex'])

    plt.figure(figsize=(16,9))
    plt.plot(PV_forecast, label='forecast')
    plt.plot(PV_oracle, linestyle='--', label='oracle')
    plt.plot(PV_worst, linestyle='-.', label='worst')
    plt.plot(PV_min, linestyle='--', color='darkgrey')
    plt.plot(PV_max, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_PV_comparison' + '.pdf')
    plt.close('all')

    FONTSIZE = 20
    plt.figure(figsize=(16,9))
    plt.plot(load_forecast, label='forecast')
    plt.plot(load_oracle, linestyle='--', label='oracle')
    plt.plot(load_worst, linestyle='-.', label='worst')
    plt.plot(load_min, linestyle='--', color='darkgrey')
    plt.plot(load_max, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_load_comparison' + '.pdf')
    plt.close('all')

    # MILP planner with forecasts
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    planner.solve()
    solution = planner.store_solution()
    # dump_file(dir=dirname, name='solution_forecasts', file=solution['x'])

    print('objective point forecasts %.2f' % (solution['obj']))

    # MILP planner with perfect forecasts -> oracle
    planner_oracle = Planner_MILP(PV_forecast=PV_oracle, load_forecast=load_oracle)
    planner_oracle.export_model(dirname + 'planner_MILP')
    planner_oracle.solve()
    solution_oracle = planner_oracle.store_solution()

    print('objectvalue oracle %.2f' % (solution_oracle['obj']))

    # MILP planner with worst
    planner_worst = Planner_MILP(PV_forecast=PV_worst, load_forecast=load_worst)
    planner_worst.solve()
    solution_worst = planner_worst.store_solution()

    print('objectvalue worst %.2f' % (solution_worst['obj']))

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_b'], label='x_b predict')
    plt.plot(solution_worst['x_b'], label='x_b worst')
    plt.plot(solution_oracle['x_b'], linestyle='--', label='x_b oracle')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('MILP formulation')
    plt.tight_layout()
    plt.savefig(dirname + day + '_x_b_comparison' + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution['y_curt'], label='y_curt predict')
    plt.plot(solution_oracle['y_curt'], linestyle='--', label='y_curt oracle')
    plt.plot(solution_worst['y_curt'], linestyle='-.', label='y_curt worst')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_y_curt_comparison' + '.pdf')
    plt.close('all')
    
    plt.figure(figsize=(16,9))
    plt.plot(solution_oracle['y_diesel'], color='firebrick', linewidth=2, label='Diesel')
    plt.plot(([hs - eg for hs, eg in zip(solution_oracle['y_discharge'], solution_oracle['y_charge'])]), color='seagreen', linewidth=2, label='BESS')
    plt.plot(solution_oracle['y_load'], color='darkorange', linewidth=2, label='load')
    plt.plot(solution_oracle['y_PV'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(solution_oracle['y_curt'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution_oracle['y_PV'], solution_oracle['y_curt'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_oracle' + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution['y_diesel'], color='firebrick', linewidth=2, label='Diesel')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_discharge'], solution['y_charge'])]), color='seagreen', linewidth=2, label='BESS')
    plt.plot(solution['y_load'], color='darkorange', linewidth=2, label='load')
    plt.plot(solution['y_PV'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(solution['y_curt'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_PV'], solution['y_curt'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_forecast' + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution_worst['y_diesel'], color='firebrick', linewidth=2, label='Diesel')
    plt.plot(([hs - eg for hs, eg in zip(solution_worst['y_discharge'], solution_worst['y_charge'])]), color='seagreen', linewidth=2, label='BESS')
    plt.plot(solution_worst['y_load'], color='darkorange', linewidth=2, label='load')
    plt.plot(solution_worst['y_PV'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(solution_worst['y_curt'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution_worst['y_PV'], solution_worst['y_curt'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_worst' + '.pdf')
    plt.close('all')

    list_x = ['forecast', 'oracle', 'worst']
    list_y = [sum(solution['y_curt']), sum(solution_oracle['y_curt']), sum(solution_worst['y_curt'])]
    plt.bar(list_x, list_y)
    plt.savefig(dirname + day + 'sum of curtailment' + '.pdf')
    plt.close('all')

    print(sum(solution_worst['su_cost']))
    print(sum(solution_worst['sd_cost']))
    print(sum(solution_worst['fuel_cost']))
    print(sum(solution_worst['OM_dg_cost']))
    print(sum(solution_worst['OM_BESS_cost']))
    print(sum(solution_worst['OM_PV_cost']))
    print(sum(solution_worst['penalty_cost']))