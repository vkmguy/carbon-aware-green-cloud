import numpy as np


"""countries for the experiment"""
COUNTRIES = ["cal", "gb", "ger", "fr"]

"""Simulation duration and intervals in simulated seconds"""
MEASUREMENT_INTERVAL = 30  # mins
ERROR_REPETITIONS = 10  # Repetitions for each run with random noise that simulates forecast errors

"""Compute node"""
CLOUD_CU = np.inf
CLOUD_WATT_PER_CU = 0.5

#%%
