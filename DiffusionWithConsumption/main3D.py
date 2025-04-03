import time

from base import np, plt
from threeD import Solver3D, FluxMap3D


syst_params = {
# Space Domain
'R_dtm' : 0.5, 'R_inf' : 200,
# Consumption time
'Tc': 1e-3,
# Number of Space grids' points
'nr': 10000
}

l = syst_params['R_inf'] - syst_params['R_dtm']

def c_const(r):
    return np.ones_like(r)

def c_layer(r):
    r0 = 2 # Starting point of the step
    l  = 5 # Length of the step
    cond = (r >= r0) & (r <= r0 + l)
    c = 3/(4*np.pi) * 1/(l**3 + 3 * l**2 * r0 + 3 * l * r0**2)
    return np.where(cond , c , 0)

S3D = Solver3D(syst_params, c_layer)

t0 = time.time()
S3D.ode.solve(optimised=False)
S3D.plot.double_plot()
n_notopt = S3D.ode.n
t = time.time() - t0
print(f"Not optimised: {t:.4f} s")

t0 = time.time()
S3D.ode.solve(optimised=True)
S3D.plot.double_plot()
n_opt = S3D.ode.n
t = time.time() - t0
print(f"Optimised: {t:.4f} s")

print(f"Error: {np.linalg.norm(n_notopt - n_opt)}")

# S3D.ode.analyt.solve()
# S3D.ode.analyt.prints_flux()

plt.show()