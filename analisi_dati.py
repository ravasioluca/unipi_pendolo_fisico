"""
Relazione di Laboratorio Unipi
Ravasio Luca e Santin Jacopo

https://github.com/ravasioluca/unipi_pendolo_fisico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

data = pd.read_csv("allegati/misurazioni.csv")
properties = pd.read_csv("allegati/properties.csv")
stick_lenght = 1.00 # meters
g = 9.8052 # Recuperato da bgi.obs-mip.fr - Bureau Gravimétrique International

def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2.0 / 12.0 + d**2.0) / (g * d))

data = data.div(properties["n_measurements"].to_numpy(), axis=1)
<<<<<<< HEAD
properties["measured_period"] = data.mean(axis=0).to_numpy()
properties["error"] = np.array(data.std(axis=0) / np.sqrt(data != np.nan).sum(axis=0).to_numpy())
# error = [max(e, 0.01) for e in error]

weighted_mean = lambda x: np.average(x, weights= 1/properties.loc[x.index]["error"].to_numpy()**2)
error_deduction = lambda x: np.sqrt(1/np.sum(1/x**2))

merged_data = properties.groupby("hole_id").agg(
    measured_period=("measured_period", weighted_mean),
    hole_distance=("hole_distance", "min"),
    error=("error", error_deduction)
).reset_index()

merged_data["d"] = np.absolute(merged_data["hole_distance"].to_numpy() - stick_lenght/2)
merged_data["expected_period"] = period_model(merged_data["d"].to_numpy(), stick_lenght)
merged_data["sigma"] = np.abs(merged_data["expected_period"] - merged_data["measured_period"]) / merged_data["error"]

print(merged_data)
=======
measured_times = data.mean(axis=0).to_numpy()
error = np.array(data.std(axis=0) / np.sqrt(data != np.nan).sum(axis=0).to_numpy())
error = [max(e, 0.01) for e in error]

properties["d"] = np.absolute(properties["hole_distance"].to_numpy() - stick_lenght/2)
expected_time = 2*math.pi * np.sqrt((stick_lenght**2.0 / 12.0 + properties["d"].to_numpy()**2) / (g * properties["d"].to_numpy()))

print(data)

print("\n\n\n")
print("Expected Time: ", *(f"{x:.4f}" for x in expected_time))
print("Measured Times: ", *(f"{x:.4f}" for x in measured_times))
print("Error: ", *(f"{x:.4f}" for x in error))
print("Actual Sigmas: ", *(f"{np.abs(x - y)/e:.4f}" for x, y, e in zip(expected_time, measured_times, error)))
>>>>>>> 124ecc70b604d4ef8f5499793b8ac1974976d40f
