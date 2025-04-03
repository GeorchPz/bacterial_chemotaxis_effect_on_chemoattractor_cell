from base import np, plt
from oneD import Solver1D, FluxMap1D

syst_params = {
'L' : 1.0,  'T' : 1.0, # Lenght & Time Domain
'Tc': 1.0,             # Consumption time
'nx': 50,   'nt': 100  # Num Spatial/Temporal points
}
L = syst_params['L']


# Initial condition for the nutrients
n0_linear = lambda x: x / L

# Bacterium distribution functions
def c_const(x):
    return np.ones_like(x)

def c_midstep(x):
    x0 = L/3 # Starting point of the step
    l  = L/3 # Length of the step
    cond = (x >= x0) & (x <= x0 + l)
    return np.where( cond , 1/l , 0)

def c_exp(x):
    a = 1
    return a/L * np.exp( a*(1-x/L) ) / (np.exp(a) - 1)


S1D = Solver1D(syst_params, c_exp, n0_linear)
S1D.pde.solve()
S1D.ode.solve()

S1D.plot.triple_plot()

plt.show()