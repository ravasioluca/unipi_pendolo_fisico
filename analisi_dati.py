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
stick_measurement_error = 0.005 # meters
g = 9.8052 # Recuperato da bgi.obs-mip.fr - Bureau Gravimétrique International

def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2.0 / 12.0 + d**2.0) / (g * d))

def diff_T_diff_d(d, l):
    return (np.pi * (d**2 - l**2 / 12))/(np.sqrt(g * d) * np.sqrt(l**2.0 / 12.0 + d**2.0) * d)

def diff_T_diff_l(d, l):
    return (np.pi * l)/(6 * np.sqrt(g * d) * np.sqrt(l**2.0 / 12.0 + d**2.0))

### PRENDIAMO I DATI
data = data.div(properties["n_measurements"].to_numpy(), axis=1)
properties["measured_period"] = data.mean(axis=0).to_numpy()
properties["error"] = np.array(data.std(axis=0) / np.sqrt(data != np.nan).sum(axis=0).to_numpy())
properties["error"] = np.maximum(properties["error"], 0.01/np.sqrt(3)/properties["n_measurements"])
# error = [max(e, 0.01) for e in error]

weighted_mean = lambda x: np.average(x, weights= 1/properties.loc[x.index]["error"].to_numpy()**2)
error_deduction = lambda x: np.sqrt(1/np.sum(1/x**2))
reduced_chi2 = lambda x: 1/(len(x)-1) * np.sum((x - weighted_mean(x))**2 / properties.loc[x.index]["error"].to_numpy()**2)

### PRENDIAMO SERIE DI MISURAZIONI DIVERSE PER LO STESSO FORO E LE UNIAMO IN UNA SINGOLA MISURAZIONE
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

### STAMPA DEL DATAFRAME FINALE
print(merged_data)

### PLOT DELLE MISURAZIONI E DEL MODELLO TEORICO
plt.plot(merged_data["hole_distance"], merged_data["measured_period"], "o", label="Misurazioni")

spazio_lineare = np.linspace(0, 0.485, 1000)
plt.plot(spazio_lineare, period_model(np.abs(spazio_lineare - stick_lenght/2), stick_lenght), label="Modello teorico", color="orange")


spazio_lineare = np.linspace(0.515, 1, 1000)
plt.plot(spazio_lineare, period_model(np.abs(spazio_lineare - stick_lenght/2), stick_lenght), label="Modello teorico", color="orange")

plt.axvline(x=stick_lenght/2, color="orange", linestyle="--", label="Centro del pendolo")
plt.margins(x=0, y=0.1)

plt.show()

### PLOT DEI RESIDUI

plt.errorbar(merged_data["hole_distance"], merged_data["measured_period"] - merged_data["expected_period"], yerr=merged_data["error"], fmt=".", label="Misurazioni con errore")

linspace = np.linspace(0, 0.4, 1000)
expected_T_error = np.sqrt(diff_T_diff_d(np.abs(linspace - stick_lenght/2), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - stick_lenght/2), stick_lenght)**2 * stick_measurement_error**2)

plt.plot(linspace, expected_T_error, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error, color="orange", linestyle="--")

linspace = np.linspace(0.6, 1, 1000)
expected_T_error = np.sqrt(diff_T_diff_d(np.abs(linspace - stick_lenght/2), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - stick_lenght/2), stick_lenght)**2 * stick_measurement_error**2)

plt.plot(linspace, expected_T_error, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error, color="orange", linestyle="--")
plt.margins(0)


plt.show()
