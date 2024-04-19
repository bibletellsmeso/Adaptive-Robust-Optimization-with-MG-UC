import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# 1. PV configuration of the NEHLA case study
PV_CAPACITY = 600  # kWp
STEP1_PERIOD_min = 15 # time resolution of the planner
STEP2_PERIOD_min = 15 # time resolution of the controller

STEP1_PERIOD_hour = STEP1_PERIOD_min / 60  # (hours)
STEP2_PERIOD_hour = STEP2_PERIOD_min / 60 # (hours)

# ------------------------------------------------------------------------------------------------------------------
# 3. BESS PARAMETERS
# PV_BESS_ratio = 100 # 100 * (BATTERY_CAPACITY / PV_CAPACITY) (%)
# PV_BATTERY_CAPACITY = PV_BESS_ratio / 100 # (kWh)

BATTERY_CAPACITY = 567 # (kWh)
BATTERY_POWER = 250 # (kW)
SOC_INI = 283.5 # (kWh)
SOC_END = SOC_INI # (kWh)

SOC_MAX = 453.6 # (kWh)
SOC_MIN = 113.4 # (kWh)

CHARGE_EFFICIENCY = 0.93 # (%)
DISCHARGE_EFFICIENCY = 0.93 # (%)
CHARGING_POWER = BATTERY_CAPACITY # (kW)
DISCHARGING_POWER = BATTERY_CAPACITY # (kW)
HIGH_SOC_PRICE = 0 # (euros/kWh) Fee to use the BESS

# ------------------------------------------------------------------------------------------------------------------

diesel_params = {"diesel_min": 200, # (kW)
                 "diesel_max": 750, # (kW)
                 "ramp_up": 150, # (kW)
                 "ramp_down": 150,
                 "p_rate": 80} # (kW)

bess_params = {"BESS_capacity": BATTERY_CAPACITY,  # (kWh)
               "soc_min": SOC_MIN,  # (kWh)
               "soc_max": SOC_MAX,  # (kWh)
               "soc_ini": SOC_INI,  # (kWh)
               "soc_end": SOC_END,  # (kWh)
               "charge_eff": CHARGE_EFFICIENCY,  # (/)
               "discharge_eff": DISCHARGE_EFFICIENCY,  # (/)
               "charge_power": BATTERY_POWER,  # (kW)
               "discharge_power": BATTERY_POWER,  # (kW)
               "HIGH_SOC_PRICE": 0}  # (euros/kWh)

cost_params = {"cost_start_up": 5,
               "cost_shut_down": 5,
               "cost_of_fuel": 6.24,
               "a_of_dg": 0.164,
               "b_of_dg": 0.046,
               "O&M_of_dg": 0.1,
               "O&M_of_BESS": 0.05,
               "O&M_of_PV": 0.01,
               "penalty_of_PV" : 0.5}

PARAMETERS = {}
PARAMETERS["period_hours"] = STEP1_PERIOD_min / 60  # (hours)
PARAMETERS['PV_capacity'] = PV_CAPACITY
PARAMETERS['cost'] = cost_params
PARAMETERS['Diesel'] = diesel_params
PARAMETERS['BESS'] = bess_params