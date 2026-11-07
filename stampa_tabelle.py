import pandas as pd

data = pd.read_csv("allegati/misurazioni.csv")
properties = pd.read_csv("allegati/properties.csv")

print(properties[["serie_associata", "hole_distance", "n_measurements"]].to_latex(index=False, float_format="%.3f"))