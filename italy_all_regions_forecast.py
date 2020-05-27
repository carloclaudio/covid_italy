import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy import integrate, optimize
from scipy.optimize import differential_evolution
import warnings




cov_ = pd.read_csv('dpc-covid19-ita-regioni.csv',parse_dates=['data'])
print(cov_.head())
initial_day=cov_['data'][0]
date_a=pd.to_datetime(initial_day)

def gauss(x, *p):
	a,b,c ,d= p
	y = (a)*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) +d
	return y

def lognorm(x, *p):
	a,b,c ,d= p
	y = (a)*np.exp(-np.power((np.log(x) - b), 2.)/(2. * c**2.)) +d
	return y




regioni = ["Sicilia","Abruzzo","Basilicata","P.A. Bolzano","Calabria","Campania","Emilia-Romagna","Friuli Venezia Giulia",
"Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna","Toscana","P.A. Trento",
"Umbria","Valle d'Aosta","Veneto"]

x_large=np.linspace(1, 120, 120)



########################################

for regione in regioni:
    print(regione)

    cov_filter=cov_['denominazione_regione']==regione
    cov=cov_[cov_filter]
    #print(cov)

    N=cov.shape[0]
    #print(N)
    L=N
    xdata = np.linspace(1, N, N)
    ydata=cov['totale_positivi']

    inizio_curva=0
    y_deceduti=cov['deceduti']
    if 1>0:
       # print(xdata)
       # print(ydata)
        while ydata.values[inizio_curva]<0:
            inizio_curva+=1
       # print(inizio_curva)
        
    #print(type(ydata.values))



    ydata = np.array(ydata, dtype=int)
    xdata = np.array(xdata, dtype=int)



    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = gauss(xdata[inizio_curva:L], *parameterTuple)
        return np.sum((ydata[inizio_curva:L] - val) ** 2.0)

    def generate_Initial_Parameters():
        minY = min(xdata[inizio_curva:L])
        maxY = max(ydata[inizio_curva:L])

        parameterBounds = []
        parameterBounds.append([-10000, 10000.0]) # search bounds for a
        parameterBounds.append([-10000, 10000.0]) # search bounds for b
        parameterBounds.append([-10000, 10000.0]) # search bounds for b
        parameterBounds.append([-10000, 10000]) # search bounds for offset

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=4)
        return result.x

    geneticParameters = generate_Initial_Parameters()
    #print(geneticParameters)

    if regione in ['Abruzzo','Calabria','Emilia-Romagna','Friuli Venezia Giulia','P.A. Bolzano','Basilicata']:
        geneticParameters = [1,1,1,1]

    popt2, pcov2 = optimize.curve_fit(gauss, xdata[inizio_curva:L]-inizio_curva, ydata[inizio_curva:L],p0=geneticParameters,maxfev=100000)
    print(popt2)
    y_gauss =  gauss(x_large, *popt2)

    popt4, pcov4 = optimize.curve_fit(lognorm, xdata[inizio_curva:L], ydata[inizio_curva:L],p0=[1,1,1,1],maxfev=100000)
    y_log =  lognorm(x_large, *popt4)
    print(popt4)
    
    def find10cases(x):
        return gauss(x,*popt2)-30


    def find10cases_log(x):
        return lognorm(x,*popt4)-30


    rmse_gauss=mean_squared_error(ydata[inizio_curva:L], y_gauss[inizio_curva:L])
    rmse_log=mean_squared_error(ydata[inizio_curva:L], y_log[inizio_curva:L])


    plt.plot(xdata, ydata,  label='Tot no of actual positives')
    

    
    if rmse_gauss < rmse_log :
        plt.plot(x_large+inizio_curva, y_gauss, color = 'orange', label='Fit gauss')
        x0 = optimize.fsolve(find10cases, 90)
        print("Days when total cases reach 30 for gauss model: "+str(x0))
        print ("MSE gauss:"+str(rmse_gauss))
        date_b=date_a+pd.DateOffset(days=int(x0))
        print("Day is: "+str(date_b))

    else:
        plt.plot(x_large, y_log, color = 'purple', label='Fit Lognormal')
        x1 = optimize.fsolve(find10cases_log, 90)
        print("Days when total cases reach 30 for log model: "+str(x1))
        print ("MSE lognorm:"+str(rmse_log))
        date_b=date_a+pd.DateOffset(days=int(x1))
        print("Day is: "+str(date_b))

    
    plt.xlabel('Days since 23 of February')
    plt.ylabel('Cases in '+str(regione))
    plt.legend()
    plt.savefig("grafici/"+regione+".png",dpi=200,bbox_inches='tight')
    plt.clf()
    #plt.show()


