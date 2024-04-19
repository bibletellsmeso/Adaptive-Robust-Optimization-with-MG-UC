import os
import time
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
from Params import PARAMETERS
from Data_read import *

class CCG_MP():
    """
    CCG = Column-and-Constraint Generation.
    MP = Master Problem of the CCG algorithm.
    The MP is a Linear Programming.

    :ivar nb_periods: number of market periods (-)
    
    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array=None, load_forecast:np.array=None):
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
        self.curt = data.PV_pred

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
        model = gp.Model("MP")

        # -------------------------------------------------------------------------------------------------------------
        # 2.1 Create First-stage variables -> x
        x_su = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_su') # start-up
        x_sd = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_sd') # shut-down
        x_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_b') # on/off binary
        theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name="theta") # objective

        # -------------------------------------------------------------------------------------------------------------
        # 2.2 Create objective
        model.setObjective(theta, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 2.3 Create constraints
        # Diesel strat-up, shut-down cst
        model.addConstrs((x_su[i] + x_sd[i] <= 1 for i in self.t_set), name='c_diesel_su-sd')
        # Diesel on/off cst
        model.addConstrs((x_su[i] - x_sd[i] == x_b[i] - x_b[i-1] for i in range(1, self.nb_periods)), name='c_x_b')

        # -------------------------------------------------------------------------------------------------------------
        # 3. Store variables
        self.allvar = dict()
        self.allvar['x_su'] = x_su
        self.allvar['x_sd'] = x_sd
        self.allvar['x_b'] = x_b
        self.allvar['theta'] = theta

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def update_MP(self, PV_trajectory:np.array, load_trajectory:np.array, iteration:int):

        """
        Add the second-stage variables at CCG iteration i.
        :param MP: MP to update in the CCG algorithm.
        :param PV_trajectory: PV trajectory computed by the SP at iteration i.
        :param iteration: update at iteration i.
        :return: the model is directly updated
        """
        # -------------------------------------------------------------------------------------------------------------
        # 4.1 Second-stage variables -> y
        # Diesel power (kW)
        y_diesel = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_diesel_" + str(iteration))
        # State of charge of the battery (kWh)
        y_s = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s_" + str(iteration))
        # Charging power (kW)
        y_charge = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge_" + str(iteration))
        # Discharging power (kW)
        y_discharge = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge_" + str(iteration))
        # binary variable -> y_b = 1: charge / y_b = 0: discharge
        y_b = self.model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b_" + str(iteration))
        # PV generation (kW)
        y_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV_" + str(iteration))
        # PV curtailment (kW)
        y_curt = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_" + str(iteration))
        # Load load (kW)
        y_load = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load_" + str(iteration))
        # PV piecewise curtailment (kW)
        # y_curt_pie = self.model.addVars(self.N_PWL, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_pie_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # diesel start_up cost
        su_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='dg_su_cost_' + str(iteration))
        # diesel shut_down cost
        sd_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='dg_sd_cost_' + str(iteration))
        # diesel fuel cost
        fuel_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="dg_cost_" + str(iteration))
        # diesel O&M cost
        OM_dg_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_dg_cost_" + str(iteration))
        # BESS O&M cost
        OM_BESS_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_BESS_cost_" + str(iteration))
        # PV O&M cost
        OM_PV_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_PV_cost_" + str(iteration))
        # PV penalty cost
        penalty_cost = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="PV_penalty_cost_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.2 Add the constraint related to the objective
        # objective
        objective = gp.quicksum(su_cost[i] + sd_cost[i] + fuel_cost[i] + OM_dg_cost[i] + OM_BESS_cost[i] + OM_PV_cost[i] + penalty_cost[i] for i in self.t_set)
        # theta = MP.model.getVarByname() = only theta variable of the MP model
        self.model.addConstr(self.model.getVarByName('theta') >= objective, name='theta_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.3 Add the constraint related to the feasbility domain of the secondstage variables -> y

        # 4.3.1 cost cst
        # diesel start_up cost
        self.model.addConstrs(su_cost[i] == self.cost_su * self.model.getVarByName('x_su[' + str(i) + ']') for i in self.t_set)
        # diesel shut_down cost
        self.model.addConstrs(sd_cost[i] == self.cost_sd * self.model.getVarByName('x_sd[' + str(i) + ']') for i in self.t_set)
        # diesel fuel cost
        self.model.addConstrs(fuel_cost[i] == self.period_hours * (self.cost_fuel * ((self.cost_a * y_diesel[i]) + self.cost_b * self.p_rate * self.model.getVarByName('x_b[' + str(i) + ']'))) for i in self.t_set)
        # diesel O&M cost
        self.model.addConstrs(OM_dg_cost[i] == self.period_hours * self.OM_dg * y_diesel[i] for i in self.t_set)
        # BESS O&M cost
        self.model.addConstrs(OM_BESS_cost[i] == self.period_hours * self.OM_BESS * (y_charge[i] + y_discharge[i]) for i in self.t_set)
        # PV O&M cost
        self.model.addConstrs(OM_PV_cost[i] == self.period_hours * self.OM_PV * y_PV[i] for i in self.t_set)
        # PV penalty cost
        self.model.addConstrs(penalty_cost[i] == self.period_hours * self.cost_penalty * y_curt[i] for i in self.t_set)

        # 4.3.2 Diesel
        # min diesel cst: self.model.getVarByName() -> return variables of the model from name, the x_b variable are index 0 to 95
        self.model.addConstrs((y_diesel[i] >= self.diesel_min * self.model.getVarByName('x_b[' + str(i) + ']') for i in self.t_set), name='c_min_diesel_' + str(iteration))
        # max diesel cst
        self.model.addConstrs((y_diesel[i] <= self.diesel_max * self.model.getVarByName('x_b[' + str(i) + ']') for i in self.t_set), name='c_max_diesel_' + str(iteration))
        # diesel ramp down cst
        self.model.addConstrs((- y_diesel[i] + y_diesel[i-1] <= self.diesel_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-down_' + str(iteration))
        # diesel ramp up cst
        self.model.addConstrs((y_diesel[i] - y_diesel[i-1] <= self.diesel_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_diesel_ramp-up_' + str(iteration))       

        # 4.3.3 power balance equation
        self.model.addConstrs((y_diesel[i] + y_PV[i] - y_curt[i] + (y_discharge[i] - y_charge[i]) - y_load[i] == 0 for i in self.t_set), name='c_power_balance_eq_' + str(iteration))

        # 4.3.4 BESS
        # max charge cst
        self.model.addConstrs((y_charge[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_max_charge_' + str(iteration))
        # max discharge cst
        self.model.addConstrs((y_discharge[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set), name='c_max_discharge_' + str(iteration))
        # min soc cst
        self.model.addConstrs((y_s[i] >= self.soc_min for i in self.t_set), name='c_min_s_' + str(iteration))
        # min soc cst
        self.model.addConstrs((y_s[i] <= self.soc_max for i in self.t_set), name='c_max_s_' + str(iteration))
        # BESS dynamics first period
        self.model.addConstr((y_s[0] == self.soc_ini), name='c_BESS_first_period_' + str(iteration))
        # BESS dynamics from second to last periods
        self.model.addConstrs((y_s[i] - y_s[i - 1] - self.period_hours * ((self.charge_eff * y_charge[i]) - (y_discharge[i] / self.discharge_eff)) == 0
                               for i in range(1, self.nb_periods)), name='c_BESS_dynamics_' + str(iteration))
        # BESS dynamics last period
        self.model.addConstr((y_s[self.nb_periods - 1] == self.soc_end), name='c_BESS_last_period_' + str(iteration))

        # 4.3.5 PV curtailment
        self.model.addConstrs((y_curt[i] <= self.curt[i] for i in self.t_set), name='c_curtailment_' + str(iteration))
        # # PV curtailment 10 piece
        '''
        self.model.addConstr((y_curt_pie[i] <= 400/10 for i in range(10)), name='c_max_curt_piece_' + str(iteration))
        self.model.addConstr((y_curt_pie[i] >= 0 for i in range(10)), name='c_min_curt_piece_' + str(iteration))
        '''
        # 4.3.6 PV generation cst
        self.model.addConstrs((y_PV[i] == PV_trajectory[i] for i in self.t_set), name='c_PV_generation_' + str(iteration))
        # 4.3.7 load cst
        self.model.addConstrs((y_load[i] == load_trajectory[i] for i in self.t_set), name='c_load_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store the added variables to the MP in a new dict
        self.allvar['var_' + str(iteration)] = dict()
        self.allvar['var_' + str(iteration)]['y_diesel'] = y_diesel
        self.allvar['var_' + str(iteration)]['su_cost'] = su_cost
        self.allvar['var_' + str(iteration)]['sd_cost'] = sd_cost
        self.allvar['var_' + str(iteration)]['fuel_cost'] = fuel_cost
        self.allvar['var_' + str(iteration)]['OM_dg_cost'] = OM_dg_cost
        self.allvar['var_' + str(iteration)]['OM_BESS_cost'] = OM_BESS_cost
        self.allvar['var_' + str(iteration)]['OM_PV_cost'] = OM_PV_cost
        self.allvar['var_' + str(iteration)]['penalty_cost'] = penalty_cost
        self.allvar['var_' + str(iteration)]['y_s'] = y_s
        self.allvar['var_' + str(iteration)]['y_charge'] = y_charge
        self.allvar['var_' + str(iteration)]['y_discharge'] = y_discharge
        self.allvar['var_' + str(iteration)]['y_b'] = y_b
        self.allvar['var_' + str(iteration)]['y_PV'] = y_PV
        self.allvar['var_' + str(iteration)]['y_curt'] = y_curt
        self.allvar['var_' + str(iteration)]['y_load'] = y_load
        '''self.allvar['var_' + str(iteration)]['y_curt_pie'] = y_curt_pie'''

        # -------------------------------------------------------------------------------------------------------------
        # 6. Update model to implement the modifications
        self.model.update()

    # True: output a log that is generated during optimization troubleshooting to the console
    def solve(self, LogToConsole:bool=False):
        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):
        
        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            # 0 dimensional variables
            solution['theta'] = self.allvar['theta'].X
            # 1D variable
            solution['x_b'] = [self.allvar['x_b'][t].X for t in self.t_set]
            solution['x_su'] = [self.allvar['x_su'][t].X for t in self.t_set]
            solution['x_sd'] = [self.allvar['x_sd'][t].X for t in self.t_set]
            solution['obj'] = m.objVal
        else:
            print('WARNING MP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['status'] = float('nan')

        # Timing indicators
        solution['time_building'] = self.time_building_model
        solution['time_solving'] = self.time_solving_model
        solution['time_total'] = self.time_building_model + self.time_solving_model

        return solution
    
    def update_sol(self, MP_sol:dict, i:int):
        """
        Add the solution of the 1 dimensional variables at iteration i.
        :param MP_sol: solution of the MP model at iteration i.
        :param i: index of interation.
        :return: update directly the dict.
        """
        MP_status = MP_sol['status']
        if MP_status == 2 or MP_status == 9:
            MP_sol['var_' + str(i)] = dict()
            # add the solution of the 1 dimensional variables at iteration
            for var in ['y_diesel', 'su_cost', 'sd_cost', 'fuel_cost', 'y_s', 'OM_dg_cost', 'OM_BESS_cost', 'OM_PV_cost',
                        'penalty_cost', 'y_charge', 'y_discharge', 'y_PV', 'y_curt', 'y_load', 'y_b']:
                MP_sol['var_' + str(i)][var] = [self.allvar['var_' + str(i)][var][t].X for t in self.t_set]
        else:
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')

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
