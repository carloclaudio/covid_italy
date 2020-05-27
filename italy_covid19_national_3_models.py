import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy import integrate, optimize
import math
from scipy.stats import gamma
from sympy.solvers import solve
from sympy import Symbol
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
import warnings


# Read input data

cov = pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv',parse_dates=['data'])
N=cov.shape[0]
L=N
POP=6500000  #Population of Italy
xdata = np.linspace(1, N, N)
ydata=cov['totale_positivi']
ydata = np.array(ydata, dtype=float)
xdata = np.array(xdata, dtype=float)

# Mathematical models

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

def weib(x,*p):
    n,a,b=p
    return b*(a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

def lognorm(x, *p):
	a,b,c ,d= p
	y = (a)*np.exp(-np.power((np.log(x) - b), 2.)/(2. * c**2.)) +d
	#y=a /( 1+b*np.exp(-c*x))+d
	#print("y: "+str(y))
	return y

def factorial(n):
    
    fact = 1
    if int(n) >= 1:
        for i in range (1,int(n)+1):
           fact = fact * i
    return float(fact)

    
def gam(x, *p):
	alfa,beta= p
	#y = (a)*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) +d
	#y=a /( 1+b*np.exp(-c*x))
	y=((beta**alfa)*(x**(alfa-1))*np.exp(-beta*x))/factorial(alfa-1)
	#print("y: "+str(y))
	return y

def gompertz(x, *p ):
    a,b,c=p
    return a*np.exp(-np.exp(b-c*x))


I0 = ydata[0]
#print(I0)
R0 = 0
S0 = POP - I0-R0


x_large=np.linspace(1, 120, 120)

# Popt contains coefficients for SIR model
popt, pcov = optimize.curve_fit(fit_odeint, xdata[0:L], ydata[0:L])
y_sir = fit_odeint(x_large,*popt)
y_sir_complete=full_fit_odeint(xdata, *popt)
y_sir_complete = np.array(y_sir_complete, dtype=int)
#print(y_sir_complete)



popt2, pcov2 = optimize.curve_fit(gauss, xdata[0:L], ydata[0:L],p0=[1,1,1,1],maxfev=100000)
y_gauss =  gauss(x_large, *popt2)



# Genetic alghoritm

def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = gauss(xdata[0:L], *parameterTuple)
    return np.sum((ydata[0:L] - val) ** 2.0)

def generate_Initial_Parameters():
    minY = min(xdata[0:L])
    maxY = max(ydata[0:L])

    parameterBounds = []
    parameterBounds.append([-100000, 100000.0]) # search bounds for a
    parameterBounds.append([-100000, 100000.0]) # search bounds for b
    parameterBounds.append([-100000, 100000.0]) # search bounds for b
    parameterBounds.append([-100000, 100000]) # search bounds for offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=4)
    return result.x

geneticParameters = generate_Initial_Parameters()
#print(geneticParameters)


popt4, pcov4 = optimize.curve_fit(lognorm, xdata[0:L], ydata[0:L],p0=geneticParameters,maxfev=100000)
y_log =  lognorm(x_large, *popt4)

# We want to find root of equation , ie when curves reach 100 infections

def find100cases(x):
    return gauss(x,*popt2)-100


def find100cases_log(x):
    return lognorm(x,*popt4)-100


x0 = optimize.fsolve(find100cases, 100)
print("Days when total cases reach 100 for gauss model: "+str(x0))
initial_day=cov['data'][0]
date_a=pd.to_datetime(initial_day)
date_b=date_a+pd.DateOffset(days=int(x0))
print("Day is: "+str(date_b))


x1 = optimize.fsolve(find100cases_log, 100)
print("Days when total cases reach 100 for lognorm: "+str(x1))
date_c=date_a+pd.DateOffset(days=int(x1))
print("Day is: "+str(date_c))

print ("MSE gauss:"+str(mean_squared_error(ydata, y_gauss[0:L])))
print ("MSE lognorm:"+str(mean_squared_error(ydata, y_log[0:L])))
print ("MSE SIR:"+str(mean_squared_error(ydata, y_sir[0:L])))


plt.plot(xdata, ydata,label='Actual data')
plt.plot(xdata[L:], ydata[L:], 'o',color='grey')
plt.plot(x_large, y_gauss, color = 'orange',label='Gauss fit')
plt.plot(x_large, y_log, color = 'pink',label='Lognorm fit')
plt.plot(x_large, y_sir, color='green', label='SIR')
plt.xlabel("Days since 23 February")
plt.ylabel("Total no of positive cases")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)
plt.legend()
plt.show()
