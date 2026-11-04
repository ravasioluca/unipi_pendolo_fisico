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

dati = pd.read_csv("allegati/dati.csv")

g = 9.8052 # Recuperato da bgi.obs-mip.fr - Bureau Gravimétrique International

def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2.0 / 12.0 + d**2.0) / (g * d))