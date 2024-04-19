"""
Utils file containing several functions to be used.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from Params import PARAMETERS

def check_BESS(SP_primal_sol: dict):
    """
    Check if there is any simultanenaous charge and discharge of the BESS.
    :param SP_primal_sol: solution of the SP primal, dict with at least the keys arguments y_charge and y_charge.
    :return: number of simultanenaous charge and discharge.
    """
    df_check = pd.DataFrame(SP_primal_sol['y_charge'], columns=['y_charge'])
    df_check['y_discharge'] = SP_primal_sol['y_discharge']

    nb_count = 0
    for i in df_check.index:
        if (df_check.loc[i]['y_charge'] > 0) and (df_check.loc[i]['y_discharge'] > 0):
            nb_count += 1
    return nb_count

def build_point_forecast(dir: str= '/Users/Andrew/OneDrive - GIST/Code/M.S. Code/'):
    """
    Load PV dad point forecasts of VS1 and VS2.
    :return: pv_solution, pv_dad_VS1, and pv_dad_VS2
    """

    k1 = 11 # 0 or 11
    k2 = 80 # 95 or 80
    pv_dad_VS = pd.read

def dump_file(dir: str, name: str, file):
    """
    Dump a file into a picke.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir: str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file
    