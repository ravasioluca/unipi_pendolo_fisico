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

plt.rcParams.update({
    "text.usetex": True,  # Attiva l'uso di LaTeX per i testi
    "font.family": "serif",  # Consigliato con LaTeX (es. Computer Modern)
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 18,  # Aumentato per renderla più grande
    "legend.title_fontsize": 20,  # (Opzionale) Se usi un titolo nella legenda
})


data = pd.read_csv("allegati/misurazioni.csv")
properties = pd.read_csv("allegati/properties.csv")
stick_lenght = 1.00 # meters
stick_width = 0.0152 # meters 
hole_diameter = 0.0052 # meters
stick_measurement_error = 0.0005 # meters
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
properties["error"] = np.array(data.std(axis=0) / np.sqrt(data.notna().sum(axis=0).to_numpy()))
properties["error"] = np.maximum(properties["error"], 0.01/np.sqrt(12)/properties["n_measurements"])

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
merged_data["error"] = np.sqrt(merged_data["error"]**2 + (0.01/np.sqrt(12))**2)

offset = (stick_lenght * stick_width * stick_lenght /2 - np.pi * (hole_diameter/2)**2 * sum(properties["hole_distance"].unique()))/(stick_lenght * stick_width - np.pi * (hole_diameter/2)**2 * len(properties["hole_distance"].unique()))


merged_data["d"] = np.absolute(merged_data["hole_distance"].to_numpy() - offset)
merged_data["expected_period"] = period_model(merged_data["d"].to_numpy(), stick_lenght)
merged_data["sigma"] = np.abs(merged_data["expected_period"] - merged_data["measured_period"]) / merged_data["error"]
merged_data["percent_error"] = merged_data["error"] / merged_data["expected_period"] * 100

### STAMPA DEL DATAFRAME FINALE
merged_data = merged_data.set_index("hole_id")

### PLOT DELLE MISURAZIONI E DEL MODELLO TEORICO
plt.plot(merged_data["hole_distance"], merged_data["measured_period"], "o", label="Misurazioni")

spazio_lineare = np.linspace(0, 0.485, 1000)
plt.plot(spazio_lineare, period_model(np.abs(spazio_lineare - offset), stick_lenght), label="_nolegend_", color="orange")


spazio_lineare = np.linspace(0.515, 1, 1000)
plt.plot(spazio_lineare, period_model(np.abs(spazio_lineare - offset), stick_lenght), label="Modello teorico", color="orange")

plt.axvline(x=offset, color="orange", linestyle="--", label="Centro di massa")
plt.margins(x=0, y=0.1)
plt.xlabel("Distanza del foro dall'estremo (m)")
plt.ylabel("Periodo (s)")
plt.title("Periodo di oscillazione in funzione della posizione del foro")
plt.legend(loc="best")
plt.show()
plt.close()

### PLOT DEI RESIDUI

plt.errorbar(merged_data["hole_distance"], merged_data["measured_period"] - merged_data["expected_period"], yerr=merged_data["error"], fmt=".", label="Misurazioni con errore")

concatenated = np.array([])
linspace = np.linspace(0, 0.48, 1000)
expected_T_error = np.sqrt(diff_T_diff_d(np.abs(linspace - offset), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - offset), stick_lenght)**2 * (stick_measurement_error*2)**2)
concatenated = np.concatenate([concatenated, expected_T_error])

plt.plot(linspace, expected_T_error, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error, color="orange", linestyle="--")
concatenated = np.concatenate([concatenated, np.zeros(1000) + max(expected_T_error)])

linspace = np.linspace(0.52, 1, 1000)
expected_T_error = np.sqrt(diff_T_diff_d(np.abs(linspace - offset), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - offset), stick_lenght)**2 * (stick_measurement_error*2)**2)
concatenated = np.concatenate([concatenated, expected_T_error])

plt.plot(linspace, expected_T_error, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error, color="orange", linestyle="--")
plt.margins(0)

plt.fill_between(
    np.concatenate([np.linspace(0, 0.48, 1000), np.linspace(0.48, 0.52, 1000), np.linspace(0.52, 1, 1000)]),
    concatenated,
    -concatenated,
    color="orange",
    alpha=0.1,
    label="Incertezza sul valore atteso"
)
plt.xlabel("Distanza del foro dall'estremo (m)")
plt.ylabel("Residuo: misurato − atteso (s)")
plt.title("Residui assoluti")
plt.legend(loc="upper left")
plt.show()
plt.close()

### PLOT DEI RESIDUI PERCENTUALI

plt.errorbar(merged_data["hole_distance"], (merged_data["measured_period"] - merged_data["expected_period"]) / merged_data["expected_period"] * 100, yerr=merged_data["error"] / merged_data["expected_period"] * 100, fmt=".", label="Misurazioni con errore")

concatenated = np.array([])

linspace = np.linspace(0, 0.48, 1000)
expected_T_error_percentual = np.sqrt(diff_T_diff_d(np.abs(linspace - offset), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - offset), stick_lenght)**2 * (stick_measurement_error*2)**2) / period_model(np.abs(linspace - offset), stick_lenght) * 100
concatenated = np.concatenate([concatenated, expected_T_error_percentual])

plt.plot(linspace, expected_T_error_percentual, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error_percentual, color="orange", linestyle="--")

concatenated = np.concatenate([concatenated, np.zeros(1000) + max(expected_T_error_percentual)])

linspace = np.linspace(0.52, 1, 1000)
expected_T_error_percentual = np.sqrt(diff_T_diff_d(np.abs(linspace - offset), stick_lenght)**2 * stick_measurement_error**2 + diff_T_diff_l(np.abs(linspace - offset), stick_lenght)**2 * (stick_measurement_error*2)**2) / period_model(np.abs(linspace - offset), stick_lenght) * 100
concatenated = np.concatenate([concatenated, expected_T_error_percentual])
plt.plot(linspace, expected_T_error_percentual, color="orange", linestyle="--")
plt.plot(linspace, -expected_T_error_percentual, color="orange", linestyle="--")

plt.fill_between(
    np.concatenate([np.linspace(0, 0.48, 1000), np.linspace(0.48, 0.52, 1000), np.linspace(0.52, 1, 1000)]),
    concatenated,
    -concatenated,
    color="orange",
    alpha=0.1,
    label="Incertezza sul valore atteso"
)

merged_data["error_percentual"] = merged_data["error"] / merged_data["expected_period"] * 100

weighted_mean = lambda x: np.average(x, weights= 1/merged_data.loc[x.index]["error_percentual"].to_numpy()**2)
plt.axhline(y=weighted_mean((merged_data["measured_period"] - merged_data["expected_period"]) / merged_data["expected_period"] * 100), color="green", linestyle="--", label="Scostamento percentuale medio")
plt.margins(0)
plt.xlabel("Distanza del foro dall'estremo (m)")
plt.ylabel(r"Residuo percentuale (\%)")
plt.title("Residui percentuali")
plt.legend(loc="upper left")
plt.show()
plt.close()
