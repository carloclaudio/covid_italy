import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


# Step 1, load data source reading date

# File can be found here: https://github.com/pcm-dpc/COVID-19/blob/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
cov = pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv',parse_dates=['data'])
pd.set_option('display.max_rows', None)

#print(cov['deceduti'])


# Step 2, view graphical data 
ax = plt.gca()   #Get current axis
cov.plot(x='data', y='ricoverati_con_sintomi', kind='line',ax=ax)
cov.plot(x='data', y='terapia_intensiva', kind='line',color='yellow',ax=ax)
#cov.plot(x='data', y='totale_positivi', kind='line',ax=ax )
cov.plot(x='data', y='isolamento_domiciliare', kind='line',ax=ax,color='pink')
cov.plot(x='data', y='dimessi_guariti', kind='line',color='green',ax=ax)
cov.plot(x='data', y='deceduti', kind='line',color='red',ax=ax)
plt.xlabel("Time")

date_form = DateFormatter("%d-%b")
ax.xaxis.set_major_formatter(date_form)
#ax.xaxis.set_minor_locator(mdates.DayLocator())
#ax.xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()
plt.show()
