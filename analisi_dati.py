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
properties["measured_period"] = data.mean(axis=0).to_numpy()
properties["error"] = np.array(data.std(axis=0) / np.sqrt(data != np.nan).sum(axis=0).to_numpy())
# error = [max(e, 0.01) for e in error]

weighted_mean = lambda x: np.average(x, weights= 1/properties.loc[x.index]["error"].to_numpy()**2)
error_deduction = lambda x: np.sqrt(1/np.sum(1/x**2))
reduced_chi2 = lambda x: 1/(len(x)-1) * np.sum((x - weighted_mean(x))**2 / properties.loc[x.index]["error"].to_numpy()**2)

merged_data = properties.groupby("hole_id").agg(
    measured_period=("measured_period", weighted_mean),
    hole_distance=("hole_distance", "min"),
    error=("error", error_deduction),
    reduced_chi2=("measured_period", reduced_chi2)
).reset_index()

merged_data["error"] = np.maximum(np.sqrt(merged_data["reduced_chi2"]), 1) * merged_data["error"]

# merged_data["error"] = np.maximum(merged_data["error"], 0.01) -- molto controverso

merged_data["d"] = np.absolute(merged_data["hole_distance"].to_numpy() - stick_lenght/2)
merged_data["expected_period"] = period_model(merged_data["d"].to_numpy(), stick_lenght)
merged_data["sigma"] = np.abs(merged_data["expected_period"] - merged_data["measured_period"]) / merged_data["error"]

print(merged_data)
