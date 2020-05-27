import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np




# Step 1, load data source reading date


cov = pd.read_csv('dpc-covid19-ita-regioni-latest.csv',parse_dates=['data'])
cov_ordered=cov.sort_values(by=['totale_positivi'],ascending=False)

cov_max10=cov_ordered[0:10]

ds_filter = cov_max10.filter(['denominazione_regione','totale_positivi','deceduti','terapia_intensiva'])
print(ds_filter)


ax = ds_filter.plot.barh( x='denominazione_regione')
plt.gca().invert_yaxis()
plt.show()
