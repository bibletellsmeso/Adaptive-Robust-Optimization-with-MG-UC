import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
import matplotlib.pyplot as plt

from utils import read_file
from Params import PARAMETERS
from Data_read import *

class SP_primal_LP():
    """
    SP primal of the benders decomposition using gurobi.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar PV_forecast: PV forecast (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array, x_binary:np.array, x_startup:np.array, x_shutdown:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        self.PV_forecast = PV_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        self.x_b = x_binary # (on/off)
        self.x_su = x_startup
        self.x_sd = x_shutdown
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
        # 1. create model
        model = gp.Model("SP_primal_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create Second-stage variables -> y
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

        # diesel start_up cost
        su_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='dg_su_cost')
        # diesel shut_down cost
        sd_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='dg_sd_cost')
        # diesel fuel cost
        fuel_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="dg_cost")
        # diesel O&M cost
        OM_dg_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_dg_cost")
        # BESS O&M cost
        OM_BESS_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_BESS_cost")
        # PV O&M cost
        OM_PV_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_PV_cost")
        # PV penalty cost
        penalty_cost = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="PV_penalty_cost")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective
        objective = gp.quicksum(su_cost[i] + sd_cost[i] + fuel_cost[i] + OM_dg_cost[i] + OM_BESS_cost[i] + OM_PV_cost[i] + penalty_cost[i] for i in self.t_set)
        model.setObjective(objective, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create Second-stage constraints

        # 4.2.1 Diesel
        # min diesel cst
        model.addConstrs((y_diesel[i] >= self.diesel_min * self.x_b[i] for i in self.t_set), name='c_min_diesel')
        # max diesel cst
        model.addConstrs((y_diesel[i] <= self.diesel_max * self.x_b[i] for i in self.t_set), name='c_max_diesel')
        # diesel ramp down cst
        model.addConstrs((- y_diesel[i] + y_diesel[i-1] <= self.diesel_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-down')
        # diesel ramp up cst
        model.addConstrs((y_diesel[i] - y_diesel[i-1] <= self.diesel_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-up')

        # 4.2.2 cost cst
        # diesel start_up cost
        model.addConstrs((su_cost[i] == self.cost_su * self.x_su[i] for i in self.t_set), name='c_diesel_start_up')
        # diesel shut_down cost
        model.addConstrs((sd_cost[i] == self.cost_sd * self.x_sd[i] for i in self.t_set), name='c_diesel_shut_down')
        # diesel fuel cost
        model.addConstrs((fuel_cost[i] == self.period_hours * (self.cost_fuel * ((self.cost_a * y_diesel[i]) + self.cost_b * self.p_rate * self.x_b[i])) for i in self.t_set), name='c_diesel_fuel')
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

        model.addConstrs
        # 4.2.5 PV generation cst
        model.addConstrs((y_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_generation')
        # 4.2.6 load cst
        model.addConstrs((y_load[i] == self.load_forecast[i] for i in self.t_set), name='c_load')

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def solve(self, outputflag:bool=False):

        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status

        if solution['status'] == 2 or  solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            solution['obj'] = m.objVal

            varname = ['y_diesel', 'fuel_cost', 'y_s', 'OM_dg_cost', 'OM_BESS_cost', 'OM_PV_cost', 'penalty_cost', 'y_charge', 'y_discharge', 'y_PV', 'y_curt', 'y_load']
            for key in varname:
                solution[key] = []

            sol = m.getVars()
            solution['all_var'] = sol
            for v in sol:
                for key in varname:
                    if v.VarName.split('[')[0] == key:
                        solution[key].append(v.x)
        else:
            print('WARNING planner SP primal status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible.
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['obj'] = float('nan')

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

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())
    
    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/'

    PV_solution = np.array(pd.read_csv('PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
    load_solution = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
    PV_forecast = data.PV_pred
    load_forecast = data.load_egg

    x_binary = read_file(dir=dirname, name='sol_LP_x_b')
    x_startup = read_file(dir=dirname, name='sol_LP_x_su')
    x_shutdown = read_file(dir=dirname, name='sol_LP_x_sd')

    SP_primal = SP_primal_LP(PV_forecast=PV_forecast, load_forecast=load_forecast, x_binary=x_binary, x_startup=x_startup, x_shutdown=x_shutdown)
    SP_primal.export_model(dirname + 'SP_primal_LP')
    SP_primal.solve()
    solution = SP_primal.store_solution()

    print('objective SP primal %.2f' %(solution['obj']))
    
    plt.style.use(['science', 'no-latex'])
    plt.figure()
    plt.plot(solution['y_charge'], label='y_charge')
    plt.plot(solution['y_discharge'], label='y_discharge')
    plt.plot(solution['y_s'], label='y s')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x_binary, label='x binary')
    plt.plot(solution['y_diesel'], label='y diesel')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_discharge'], solution['y_charge'])]), label='BESS')
    plt.plot(solution['y_load'], label='load')
    plt.plot(solution['y_PV'], label= 'PV')
    plt.plot(solution['y_curt'], label='curt')
    plt.plot(([hs - eg for hs, eg in zip(solution['y_PV'], solution['y_curt'])]), label='PV output')
    plt.legend()
    plt.show()

    # Get dual values
    # for c in SP_primal.model.getConstrs():
    #     print('The dual value of %s : %g' % (c.constrName, c.pi))