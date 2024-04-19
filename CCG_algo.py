"""
Column-and-Constraint Generation (CCG) algorithm to solve a two-stage robust optimization problem in the microgrid scheduling.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CCG_MP import CCG_MP
from CCG_SP import CCG_SP
from Data_read import *
from root_project import ROOT_DIR
from utils import dump_file, check_BESS

from SP_primal_LP import SP_primal_LP
from planner_MILP import Planner_MILP
from Params import PARAMETERS

def ccg_algo(dir:str, tol: float, gamma_PV: int, gamma_load: int, PV_max: np.array, load_max: np.array, x_binary: np.array, x_startup: np.array, x_shutdown: np.array, solver_param: dict, day:str, log:bool=False, printconsole:bool=False, M_al_neg:float=None, M_al_pos:float=None, M_be_neg:float=None, M_be_pos:float=None):
    """
    CCG = Column-and-Constraint Generation
    Column-and-Constraint Generation algorithm.
    Iteration between the MP and SP until convergence criteria is reached.
    :param tol: convergence tolerance.
    :param gamma_PV/load: PV/load budget of uncertainty.
    :param PV/load_max/min: PV/load max/min bound of the uncertainty set (kW).
    :ivar x: diesel ON/OFF variable
    :param solver_param: Gurobi solver parameters.
    :return: the final ON/OFF and curtailment schedule when the convergence criteria is reached and some data.
    """

    # Compute the maxiaml deviation between the max/min PV/load uncertainty set bounds
    PV_pos = data.PV_pos # (kW)
    PV_neg = data.PV_neg
    load_pos = data.load_pos # (kW)
    load_neg = data.load_neg
    nb_periods = PV_pos.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    # CCG initialization: build the initial MP
    # ------------------------------------------------------------------------------------------------------------------

    # Building the MP
    MP = CCG_MP()
    MP.model.update()
    print('MP initialized: %d variables %d constraints' % (len(MP.model.getVars()), len(MP.model.getConstrs())))
    MP.export_model(dir + day + '_CCG_MP_initialized')

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop until convergence criteria is reached
    # ------------------------------------------------------------------------------------------------------------------

    if printconsole:
        print('---------------------------------CCG ITERATION STARTING---------------------------------')

    t_solve = time.time()
    objectives = []
    computation_times = []
    # measure that helps control the trade-off between solution quality and computation time in MILP or MIQP
    mipgap = []
    SP_dual_status = []
    SP_primal_status = []
    alpha_pos_list = []
    alpha_neg_list = []
    beta_pos_list = []
    beta_neg_list = []
    tolerance = 1e20
    # with CCG the convergence is stable.
    tolerance_list = [tolerance] * 2
    iteration = 1
    BESS_count_list = []
    BESS_charge_discharge_list = []
    max_iteration = 10

    while all(i < tol for i in tolerance_list) is not True and iteration < max_iteration:
        logfile = ""
        if log:
            logfile = dir + 'logfile_' + str(iteration) + '.log'
        if printconsole:
            print('i= %s solve SP dual' % (iteration))

        # ------------------------------------------------------------------------------------------------------------------
        # 1. SP part
        # ------------------------------------------------------------------------------------------------------------------

        # 1.1 Solve the SP and get the worst PV and load trajectory to add the new constraints of the MP
        SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_pos=PV_pos, PV_neg=PV_neg, load_pos=load_pos, load_neg=load_neg, x_binary=x_binary, x_startup=x_startup, x_shutdown=x_shutdown, gamma_PV=gamma_PV, gamma_load=gamma_load, M_al_neg=M_al_neg, M_al_pos=M_al_pos, M_be_neg=M_be_neg, M_be_pos=M_be_pos)
        SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
        SP_dual_sol = SP_dual.store_solution()
        SP_dual_status.append(SP_dual_sol['status'])
        mipgap.append(SP_dual.model.MIPGap)
        alpha_pos_list.append(SP_dual_sol['alpha_pos'])
        alpha_neg_list.append(SP_dual_sol['alpha_neg'])
        beta_pos_list.append(SP_dual_sol['beta_pos'])
        beta_neg_list.append(SP_dual_sol['beta_neg'])

        # 1.2 Compute the worst PV, load trajectory from the SP dual solution
        PV_worst_case_from_SP = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
        load_worst_case_from_SP = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]
        if printconsole:
            print('     i = %s : SP dual status %s solved in %.1f s MIPGap = %.6f' % (iteration, SP_dual_sol['status'], SP_dual_sol['time_total'], SP_dual.model.MIPGap))

        # 1.3 Solve the primal of the SP to check if the objecitves of the primal and dual are equal to each other
        SP_primal = SP_primal_LP(PV_forecast=PV_worst_case_from_SP, load_forecast=load_worst_case_from_SP, x_binary=x_binary, x_startup=x_startup, x_shutdown=x_shutdown)
        SP_primal.solve()
        SP_primal_sol = SP_primal.store_solution()
        SP_primal_status.append(SP_primal_sol['status'])

        if printconsole:
            print('     i = %s : SP primal status %s' % (iteration, SP_primal_sol['status']))
            print('     i = %s : SP primal %.1f $ SP dual %.1f $ -> |SP primal - SP dual| = %.2f $' % (iteration, SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))

        # 1.4 SP solved to optimality ? -> Check if there is any simultaneous charge and discharge in the SP primal solution
        if SP_primal_sol['status'] == 2 or SP_primal_sol['status'] == 9: # 2 = optimal, 9 = timelimit has been reached
            nb_count = check_BESS(SP_primal_sol = SP_primal_sol)
            if nb_count > 0:
                BESS_charge_discharge_list.append([iteration, SP_primal_sol['y_charge'], SP_primal_sol['y_discharge']])
            else:
                nb_count = float('nan')
            BESS_count_list.append(nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous charge and discharge' % (iteration, nb_count))

        # ------------------------------------------------------------------------------------------------------------------
        # 2. MP part
        # ------------------------------------------------------------------------------------------------------------------

        # Check Sub Problem status -> bounded or unbounded
        if SP_dual_sol['status'] == 2 or SP_dual_sol['status'] == 9:  # 2 = optimal, 9 = timelimit has been reached
            # Add an optimality cut to MP and solve
            MP.update_MP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
            if printconsole:
                print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
            # MP.export_model(dir + 'MP_' + str(iteration))
            if printconsole:
                print('i = %s : solve MP' % (iteration))
            MP.solve()
            MP_sol = MP.store_solution()
            MP.update_sol(MP_sol=MP_sol, i=iteration)
            if MP_sol['status'] == 3 or MP_sol['status'] == 4:
                print('i = %s : WARNING MP status %s -> Create a new MP, increase big-M value and compute a new PV trajectory from SP' % (iteration, MP_sol['status']))

                # MP unbounded of infeasible -> increase big-M's value to get another PV trajectory from the SP
                SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_pos=PV_pos, PV_neg=PV_neg, load_pos=load_pos, load_neg=load_neg, x_binary=x_binary, gamma_PV=gamma_PV, gamma_load=gamma_load, M_al_neg=M_al_neg+50, M_al_pos=M_al_pos+50, M_be_neg=M_be_neg+50, M_be_pos=M_be_pos+50)
                SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
                SP_dual_sol = SP_dual.store_solution()

                # Compute a new worst PV trajectory from the SP dual solution
                PV_worst_case_from_SP = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
                load_worst_case_from_SP = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]

                # Create a new MP
                MP = CCG_MP()
                MP.model.update()
                MP.update_MP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
                if printconsole:
                    print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
                # MP.export_model(dir + 'MP_' + str(iteration))
                if printconsole:
                    print('i = %s : solve new MP' % (iteration))
                MP.solve()
                MP_sol = MP.store_solution()
                MP.update_sol(MP_sol=MP_sol, i=iteration)

            computation_times.append([SP_dual_sol['time_total'], MP_sol['time_total']])

        else: # 4 = Model was proven to be either infeasible or unbounded.
            print('SP is unbounded: a feasibility cut is required to be added to the Master Problem')

        objectives.append([iteration, MP_sol['obj'], SP_dual_sol['obj'], SP_primal_sol['obj']])

        # ------------------------------------------------------------------------------------------------------------------
        # 3. Update: the x_binary, lower and upper bounds using the updated MP
        # ------------------------------------------------------------------------------------------------------------------

        # Solve the MILP with the worst case trajectory
        planner = Planner_MILP(PV_forecast=PV_worst_case_from_SP, load_forecast=load_worst_case_from_SP)
        planner.solve()
        sol_planner = planner.store_solution()

        # Update x variables
        x_binary = MP_sol['x_b']
        x_startup = MP_sol['x_su']
        x_shutdown = MP_sol['x_sd']
        # Update the lower and upper bounds
        # MP -> give the lower bound
        # SP -> give the upper bound
        tolerance = abs(MP_sol['obj'] - SP_dual_sol['obj'])
        print('i = %s : |MP - SP dual| = %.2f $' % (iteration, tolerance))
        abs_err = abs(MP_sol['obj'] - sol_planner['obj'])
        tolerance_list.append(tolerance)
        tolerance_list.pop(0)
        if printconsole:
            print('i = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration, MP_sol['obj'], SP_dual_sol['obj'], tolerance))
            print('i = %s : MP %.2f $ MILP %.2f $ -> |MP - MILP| = %.2f $' % (iteration, MP_sol['obj'], sol_planner['obj'], abs_err))
            print(tolerance_list)
            print('                                                                                                       ')

        iteration += 1

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop terminated
    # ------------------------------------------------------------------------------------------------------------------
    if printconsole:
        print('-----------------------------------CCG ITERATION TERMINATED-----------------------------------')
    print('Final iteration  = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration-1, MP_sol['obj'], SP_dual_sol['obj'], tolerance))

    # Export last MP
    MP.export_model(dir + day + '_MP')

    # MP.model.printStats()

    # Dump last engagement plan at iteration
    dump_file(dir=dir, name=day+'_x_b', file=x_binary)
    dump_file(dir=dir, name=day+'_x_su', file=x_startup)
    dump_file(dir=dir, name=day+'_x_sd', file=x_shutdown)

    # print T CPU
    t_total = time.time() - t_solve
    computation_times = np.asarray(computation_times)
    SP_dual_status = np.asarray(SP_dual_status)
    SP_primal_status = np.asarray(SP_primal_status)

    if printconsole:
        print('Total CCG loop t CPU %.1f min' % (t_total / 60))
        print('T CPU (s): Sup Problem max %.1f Master Problem max %.1f' % (computation_times[:, 0].max(), computation_times[:, 1].max()))
        print('nb Sup Problem status 2 %d status 9 %d' % (SP_dual_status[SP_dual_status == 2].shape[0], SP_dual_status[SP_dual_status == 9].shape[0]))

    # Store data
    objectives = np.asarray(objectives)
    df_objectives = pd.DataFrame(index=objectives[:,0], data=objectives[:,1:], columns=['MP', 'SP', 'SP_primal'])

    # Store convergence information
    conv_inf = dict()
    conv_inf['mipgap'] = mipgap
    conv_inf['computation_times'] = computation_times
    conv_inf['SP_status'] = SP_dual_status

    conv_inf['SP_primal_status'] = SP_primal_status
    conv_inf['alpha_pos'] = alpha_pos_list
    conv_inf['alpha_neg'] = alpha_neg_list
    conv_inf['beta_pos'] = beta_pos_list
    conv_inf['beta_neg'] = beta_neg_list
    conv_inf['BESS_count'] = BESS_count_list
    conv_inf['BESS_charge_discharge'] = BESS_charge_discharge_list

    return x_binary, x_startup, x_shutdown, df_objectives, conv_inf

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 20

# NB periods
nb_periods = 96

# Solver parameters
solver_param = dict()
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1

# Convergence threshold between MP and SP objectives
conv_tol = 5
printconsole = True

# Select the day
day_list = ['2018-07-04']
day = day_list[0]
day = '2018-07-04'

# --------------------------------------
# Static RO parameters: [q_min, gamma]
GAMMA_PV = 96 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
GAMMA_LOAD = 96
#--------------------------------------
# warm_start
M_al_neg = M_al_pos = M_be_neg = M_be_pos = 1

# quantile from NE or LSTM
PV_Sandia = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/two-stage UC by CCG/export_CCG/'
    if PV_Sandia:
        dirname += 'PV_Sandia/'
        pdfname = str(PV_Sandia) + '_' + str(GAMMA_PV) + '_' + str(GAMMA_LOAD)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    print('-----------------------------------------------------------------------------------------------------------')
    if PV_Sandia:
        print('CCG: day %s gamma_PV %s gamma_load %s' % (day, GAMMA_PV, GAMMA_LOAD))
    print('-----------------------------------------------------------------------------------------------------------')

    # Load data
    PV_oracle = data.PV_oracle
    load_oracle = data.load_oracle
    PV_forecast = data.PV_pred
    load_forecast = data.load_egg

    # plot style
    plt.style.use(['science', 'no-latex'])
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # PLOT quantile, point, and observations
    x_index = [i for i in range(0, nb_periods)]
    plt.figure(figsize=(16,9))
    plt.plot(x_index, PV_forecast, 'b', linewidth=2, label='prediction')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + str(day) + '_TSRO_forecast.pdf')
    plt.close('all')

    # Store the forecast into a dict
    PV_forecast_dict = dict()
    PV_forecast_dict['forecast'] = PV_forecast
    PV_forecast_dict['oracle'] = PV_oracle

    # Compute the PV, load min and max 
    PV_min = data.PV_min
    PV_max = data.PV_max
    load_min = data.load_min
    load_max = data.load_max

    # Max deviation between PV, load max and min bounds of the uncertainty set
    PV_pos = data.PV_pos
    PV_neg = data.PV_neg
    load_pos = data.load_pos
    load_neg = data.load_neg
    
    # Compute the starting point for the first MP = day-ahead planning from the PV using the MILP
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    planner.solve()
    sol_planner_ini = planner.store_solution()
    x_b_ini = sol_planner_ini['x_b']
    x_su_ini = sol_planner_ini['x_su']
    x_sd_ini = sol_planner_ini['x_sd']

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop
    # ------------------------------------------------------------------------------------------------------------------
    final_x_b, final_x_su, final_x_sd, df_objectives, conv_inf = ccg_algo(dir=dirname, tol=conv_tol, gamma_PV=GAMMA_PV, gamma_load=GAMMA_LOAD, PV_max=PV_max, load_max=load_max, x_binary=x_b_ini, x_startup=x_su_ini, x_shutdown=x_sd_ini, solver_param=solver_param, day=day, printconsole=printconsole, M_al_neg=M_al_neg, M_al_pos=M_al_pos, M_be_neg=M_be_neg, M_be_pos=M_be_pos)
    df_objectives.to_csv(dirname + day + 'obj_MP_SP_' + '.csv')

    print('-----------------------------------------------------------------------------------------------------------')
    print('CCG: %s gamma_PV %s gamma_load %s' %(day, GAMMA_PV, GAMMA_LOAD))
    print('-----------------------------------------------------------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Get the final worst case PV generation trajectory computed by the Sub Problem
    # ------------------------------------------------------------------------------------------------------------------

    # Get the worst case related to the last engagement plan by using the Sub Problem dual formulation
    SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_pos=PV_pos, PV_neg=PV_neg, load_pos=load_pos, load_neg=load_neg, x_binary=final_x_b, x_startup=final_x_su, x_shutdown=final_x_sd, gamma_PV=GAMMA_PV, gamma_load=GAMMA_LOAD, M_al_neg=M_al_neg, M_al_pos=M_al_pos, M_be_neg=M_be_neg, M_be_pos=M_be_pos)
    SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
    SP_dual_sol = SP_dual.store_solution()
    # Compute the worst PV, load path from the SP dual solution
    PV_worst_case = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
    dump_file(dir=dirname, name=day + '_PV_worst_case', file=PV_worst_case)
    load_worst_case = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]
    dump_file(dir=dirname, name=day + '_load_worst_case', file=load_worst_case)

    # Check if the worst PV, load path is on the extreme quantile
    # if sum(SP_dual_sol['epsilon_pos'] + SP_dual_sol['epsilon_neg']) == GAMMA_PV:
    #     print('Worst PV path is the extreme')
    # else:
    print('%d PV points on upper boundary, %d points on lower boundary, %d points on median' % (sum(SP_dual_sol['epsilon_pos']), sum(SP_dual_sol['epsilon_neg']), GAMMA_PV - sum(SP_dual_sol['epsilon_pos'] + SP_dual_sol['epsilon_neg'])))

    # if sum(SP_dual_sol['delta_pos'] + SP_dual_sol['delta_neg']) == GAMMA_LOAD:
    #     print('Worst load path is the extreme')
    # else:
    print('%d load points on upper boundary, %d points on lower boundary, %d points on median' % (sum(SP_dual_sol['delta_pos']), sum(SP_dual_sol['delta_neg']), GAMMA_LOAD - sum(SP_dual_sol['delta_pos'] + SP_dual_sol['delta_neg'])))

    # ------------------------------------------------------------------------------------------------------------------
    # Second-stage variables comparison: y_diesel, s, etc
    # ------------------------------------------------------------------------------------------------------------------

    # Use the SP primal (SP worst case dispatch max min formulation) to compute the dispatch variables related to the last CCG engagement computed by the MP
    # Use the worst case dispatch to get the equivalent of the max min formulation
    SP_primal = SP_primal_LP(PV_forecast=PV_worst_case, load_forecast=load_worst_case, x_binary=final_x_b, x_startup=final_x_su, x_shutdown=final_x_sd)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # Check if there has been any simultanenaous charge and discharge during all CCG iterations
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Check if there is any simultaneous charge and discharge at the last CCG iteration
    nb_count = check_BESS(SP_primal_sol=SP_primal_sol)
    print('CCG last iteration %d simultaneous charge and discharge' % (nb_count))

    # 2. Check if there is any simultaneous charge and discharge over all CCG iteration
    # check if there is nan value (meaning during an iteration the SP primal has not been solved because infeasible, etc)
    BESS_count = conv_inf['BESS_count']
    if sum(np.isnan(BESS_count)) > 0:
        print('WARNING %s nan values' %(sum(np.isnan(conv_inf['BESS_count']))))
    # “python list replace nan with 0” Code
    BESS_count = [0 if x != x else x for x in BESS_count]

    print('%d total simultaneous charge and discharge over all CCG iterations' % (sum(BESS_count)))
    if sum(conv_inf['BESS_count']) > 0:
        plt.figure(figsize=(16,9))
        plt.plot(conv_inf['BESS_count'], 'k', linewidth=2, label='BESS_count')
        plt.ylim(0, max(conv_inf['BESS_count']))
        plt.xlabel('iteration $j$', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.legend()
        plt.savefig(dirname + day + '_BESS_count_' + pdfname + '.pdf')
        plt.close('all')

        # Plot at each iteration where there has been a simultaneous charge and discharge
        for l in conv_inf['BESS_charge_discharge']:
            plt.figure(figsize = (8,6))
            plt.plot(l[1], linewidth=2, label='charge')
            plt.plot(l[2], linewidth=2, label='discharge')
            plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
            plt.ylabel('kW', fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            plt.title('simultaneous charge discharge at iteration %s' %(l[0]))
            plt.tight_layout()
            plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # Check CCG convergence by computing the planning for the PV worst trajectory from CCG last iteration
    # ------------------------------------------------------------------------------------------------------------------
    planner = Planner_MILP(PV_forecast=PV_worst_case, load_forecast=load_worst_case)
    planner.solve()
    sol_planner = planner.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # First-stage variables comparison: x and objectives
    # ------------------------------------------------------------------------------------------------------------------

    # Compute solution with the oracle
    planner = Planner_MILP(PV_forecast=PV_oracle, load_forecast=load_oracle)
    planner.solve()
    sol_oracle = planner.store_solution()

    # Compute solution with the point-forecasts
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    planner.solve()
    sol_forecast = planner.store_solution()

    plt.figure(figsize=(8,6))
    plt.plot(final_x_b, 'b', linewidth=2, label='x_b RO')
    # plt.plot(sol_oracle['x_b'], marker='x', linewidth=2, label='x_b perfect')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_x_b_' + pdfname + '.pdf')
    plt.savefig(dirname + '_x_binary.png', dpi=600)
    plt.close('all')

    # Convergence plot
    error_MP_SP = np.abs(df_objectives['MP'].values - df_objectives['SP'].values)
    error_SP = np.abs(df_objectives['SP'].values - df_objectives['SP_primal'].values)

    plt.figure(figsize = (8,6))
    plt.plot(error_MP_SP, marker=10, markersize=10, linewidth=2, label='|MP - SP dual| $')
    plt.plot(error_SP, marker=11, markersize=10, linewidth=2, label='|SP primal - SP dual| $')
    plt.plot(100 * np.asarray(conv_inf['mipgap']), label='SP Dual mipgap %')
    plt.xlabel('Iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('Gap', fontsize=FONTSIZE)
    plt.ylim(-1, 10)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + 'error_conv_' + pdfname + '.pdf')
    plt.savefig(dirname + '_error_conv.png', dpi=600)
    plt.close('all')

    plt.figure(figsize = (8,6))
    plt.plot(df_objectives['MP'].values, marker=10, markersize=10, linewidth=2, label='$MP^j$')
    plt.plot(df_objectives['SP'].values, marker=11, markersize=10, linewidth=2, label='$SP^j$')
    plt.plot(df_objectives['SP_primal'].values, marker='x', markersize=10, linewidth=2, label='SP primal')
    plt.hlines(y=sol_planner['obj'] + conv_tol, xmin=1, xmax=len(df_objectives['SP'].values), colors='k',
               linestyles=':', linewidth=1, label='$MILP^J \pm\ttolerance$')
    plt.hlines(y=sol_planner['obj'], xmin=1, xmax=len(df_objectives['SP'].values), linewidth=2, label='$MILP^J$')
    plt.hlines(y=sol_planner['obj'] - conv_tol, xmin=1, xmax=len(df_objectives['SP'].values), colors='k',
               linestyles=':', linewidth=1)
    plt.xlabel('Iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('Cost($)', fontsize=FONTSIZE)
    plt.ylim(round(sol_planner['obj'] * 1.05, 0), round(sol_planner['obj'] * 0.95, 0))
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, ncol=2, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + 'convergence_' + pdfname + '.pdf')
    plt.savefig(dirname + '_convergence.png', dpi=600)
    plt.close('all')

    print('')
    print('-----------------------CHECK COLUMN AND CONSTRAINT GENERATION CONVERGENCE-----------------------')
    print('Final iteration %s |MP - SP dual| %.2f $' % (
    len(df_objectives), abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1])))
    print('SP primal %.2f $ SP dual %.2f $ -> |SP primal - SP dual| = %.2f' % (
    SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))
    err_planner_CCG = abs(df_objectives['MP'].values[-1] - sol_planner['obj'])
    print('MILP planner %.2f $ MP CCG %.2f $ -> |MILP planner - MP CCG| = %.2f' % (
    sol_planner['obj'], df_objectives['MP'].values[-1], err_planner_CCG))

    if err_planner_CCG > conv_tol:
        print('-----------------------WARNING COLUMN AND CONSTRAINT GENERATION IS NOT CONVERGED-----------------------')
        print('abs error %.4f $' % (err_planner_CCG))
    else:
        print('-----------------------COLUMN AND CONSTRAINT GENERATION IS CONVERGED-----------------------')
        print('CCG is converged with |MILP planner - MP CCG| = %.4f $' % (err_planner_CCG))

    plt.figure(figsize=(16,9))
    plt.plot(PV_worst_case, color='crimson', marker="o", markersize=8, linewidth=2, label='PV trajectory')
    plt.plot(SP_primal_sol['y_curt'], color='darkviolet', marker="d", markersize=5, zorder=3, linewidth=2, label='PV curtailment')
    plt.plot(x_index, PV_forecast, 'royalblue', marker="s", linewidth=2, label='PV predection')
    plt.plot(PV_max, 'steelblue', linestyle='--', label='Upper bound', zorder=1)
    plt.plot(PV_min, 'steelblue', linestyle='--', label='Lower bound', zorder=1)
    plt.fill_between(range(96), PV_max, PV_min, color='lightsteelblue', alpha=0.6)	
    plt.xlabel('Time(h)', fontsize=FONTSIZE)
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_PV_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + '_PV_trajectory', dpi=600)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(load_worst_case, color='crimson', marker="o", markersize=8, linewidth=2, label='load trajectory')
    plt.plot(x_index, load_forecast, color='orange', marker="s", linewidth=2, label='load prediction')
    plt.plot(load_min, color='goldenrod', linestyle='--', label='Upper bound', zorder=1)
    plt.plot(load_max, color='goldenrod', linestyle='--', label='Lower bound', zorder=1)
    plt.fill_between(range(96), load_min, load_max, color='navajowhite', alpha=0.6)
    plt.xlabel('Time(h)', fontsize=FONTSIZE)
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_load_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + '_load_trajectory', dpi=600)
    plt.close('all')

    plt.figure(figsize = (8,6))
    plt.plot(sol_forecast['y_s'], 'royalblue', marker="s", markersize=4, linewidth=2, label='SOC nominal')
    plt.plot(SP_primal_sol['y_s'], 'crimson', marker="o", markersize=4, linewidth=2, label='SOC RO')
    # plt.plot(sol_oracle['y_s'], 'r', linewidth=2, label='SOC perfect')
    plt.xlabel('Time(h)', fontsize=FONTSIZE)
    plt.ylabel('kWh', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.ylim(-10, PARAMETERS['BESS']['BESS_capacity'])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_SOC_' + pdfname + '.pdf')
    plt.savefig(dirname + '_SOC', dpi=600)
    plt.close('all')
    
    plt.figure(figsize=(16,9))
    plt.plot(SP_primal_sol['y_diesel'], color='firebrick', marker="s", markersize=5, zorder=3, linewidth=2, label='Diesel Gen.')
    plt.plot(SP_primal_sol['y_load'], color='orange', zorder=3, linewidth=2, label='load')
    plt.plot(SP_primal_sol['y_PV'], color='royalblue', alpha=0.8, zorder=4, linewidth=2, label= 'PV generation')
    # plt.plot(SP_primal_sol['y_curt'], color='darkviolet', marker="d", markersize=5, zorder=3, linewidth=2, label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(SP_primal_sol['y_PV'], SP_primal_sol['y_curt'])]), color='mediumblue', marker="o", markersize=5, zorder=3, linewidth=2, label='PV output')
    plt.plot(([hs - eg for hs, eg in zip(SP_primal_sol['y_discharge'], SP_primal_sol['y_charge'])]), color='green', marker="p", markersize=5, zorder=3, linewidth=2, label='BESS')
    plt.plot(PV_max, color='steelblue', alpha=0.6, linestyle='--', label='Bound', linewidth=2, zorder=1)
    plt.plot(PV_min, color='steelblue', alpha=0.6, linestyle='--', label='Bound', linewidth=2, zorder=1)
    # plt.fill_between(range(96), PV_max, PV_min, color='c', alpha=0.2)	
    plt.plot(load_min, color='goldenrod', alpha=0.6, linestyle='--', label='Bound', linewidth=2, zorder=1)
    plt.plot(load_max, color='goldenrod', alpha=0.6, linestyle='--', label='Bound', linewidth=2, zorder=1)
    # plt.fill_between(range(96), load_min, load_max, color='orange', alpha=0.2)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('Time(h)', fontsize=FONTSIZE)
    plt.ylabel('Power(kW)', fontsize=FONTSIZE)
    plt.ylim(-250, PARAMETERS['PV_capacity'])
    plt.legend(('Diesel Gnerator Output', 'Demand', 'PV Generator Output', 'PV Output', 'BESS'), ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=14)
    plt.savefig(dirname + day + '_Result_' + pdfname + '.pdf', dpi=600)
    plt.savefig(dirname + day + '_Result.png', dpi=600)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(SP_primal_sol['y_charge'], linewidth=2, label='charge')
    plt.plot(SP_primal_sol['y_discharge'], linewidth=2, label='discharge')
    plt.ylim(0, PARAMETERS['BESS']['BESS_capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_charge_discharge_' + pdfname + '.pdf')
    plt.close('all')

    PV_worst = np.array(SP_primal_sol['y_PV'])
    load_worst = np.array(SP_primal_sol['y_load'])
    fmt = '%.18e'
    data = np.column_stack((PV_worst, load_worst.flatten()))
    np.savetxt('worst.csv', data, delimiter=',', header='PV_worst, load_worst', comments='', fmt='%.2f')