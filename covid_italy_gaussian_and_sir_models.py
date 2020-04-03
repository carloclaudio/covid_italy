import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy import integrate, optimize



cov = pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv',parse_dates=['data'])
##                              dataset available at:
##                              https://github.com/pcm-dpc/COVID-19/tree/master/dati-andamento-nazionale


N=cov.shape[0]                  #Number of days with historical data
L=30                            # We will consider only L days out of N, to build our models.
                                # This way we can see how they predict future data
                                
POP=46500000                    # Total population of Italy
xdata = np.linspace(1, N, N)
ydata=cov['totale_positivi']    #Total number of confirmed cases, it doesn't include deaths or recovered 
ytotcas=cov['totale_casi']      #Total number of  cases, it  include deaths and recovered 


ydata = np.array(ydata, dtype=float)
xdata = np.array(xdata, dtype=float)

def sir_model(y, x, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / POP
    dIdt = beta * S * I / POP - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]

def full_fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))

def gauss(x, *p):
	a,b,c ,d= p
	y = (a)*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) +d
	return y



I0 = 221       # The first day we have 221 infected people
R0 = 8         # The first day we have 1 death and 7 recovered persons
S0 = POP - I0-R0



x_large=np.linspace(1, 100, 100)    # Our simulation will forecast 100 days

popt, pcov = optimize.curve_fit(fit_odeint, xdata[0:L], ydata[0:L])
y_sir = fit_odeint(x_large,*popt)

y_sir_complete=full_fit_odeint(xdata, *popt)
y_sir_complete = np.array(y_sir_complete, dtype=int)
print(y_sir_complete)
# According to our SIR model, the 2n day we should have 283 infections, and     1620 deaths or recovered people.
#This is really impossible! Actual data is I=311, and R=11

########################################

popt2, pcov2 = optimize.curve_fit(gauss, xdata[0:L], ydata[0:L],p0=[1.1,1.1,1.1,1.1],maxfev=100000)
y_gauss =  gauss(x_large, *popt2)


plt.plot(xdata[0:L], ydata[0:L], 'o', label='Actual data used for fit', color='pink')
plt.plot(xdata[L:N], ydata[L:N], 'o', label='Actual data (all)', color='grey')


plt.plot(x_large, y_gauss, color = 'orange', label='Gaussian fit')
plt.plot(x_large, y_sir, color='green',label='SIR fit')
plt.legend()

plt.show()
