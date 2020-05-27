import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np



# Define populations for each region

regioni = {'denominazione_regione':  ["Sicilia","Abruzzo","Basilicata","P.A. Bolzano","Calabria","Campania","Emilia-Romagna","Friuli Venezia Giulia",
                    "Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna","Toscana","P.A. Trento",
                        "Umbria","Valle d'Aosta","Veneto"],
        'Pop': [4999891,1311580,562869,533050,1947131,5801692,4459477 , 1215220 , 5879082,1550640, 10060574,1525271, 305617, 4356406, 4029053,
                1639591, 3729641,541380,882015,125666, 4905854  ]
        }



regioni_df = pd.DataFrame(regioni, columns = ['denominazione_regione', 'Pop'])


# Step 1, load data source reading date


cov = pd.read_csv('dpc-covid19-ita-regioni-latest.csv',parse_dates=['data'])
cov2=pd.merge(regioni_df,cov, on='denominazione_regione')

sel = cov2[["denominazione_regione","Pop","totale_positivi",'deceduti','terapia_intensiva']]


new_df = sel.copy()
new_df["totale_positivi_norm"]=new_df["totale_positivi"]/new_df["Pop"]*100000
new_df["deceduti_norm"]=new_df["deceduti"]/new_df["Pop"]*100000
new_df["terapia_intensiva_norm"]=(new_df["terapia_intensiva"]/new_df["Pop"])*1000000


pd.set_option('display.max_columns', None)


cov_ordered=new_df.sort_values(by=['totale_positivi_norm'],ascending=False)

cov_max10=cov_ordered[0:10]
final = cov_max10.filter(['denominazione_regione','totale_positivi_norm','deceduti_norm','terapia_intensiva_norm'])
print(final)



ax = final.plot.barh( x='denominazione_regione')
plt.gca().invert_yaxis()
plt.show()
