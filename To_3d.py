# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:58:09 2024

@author: Jorge Pottiez López-Jurado
"""

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from IPython import get_ipython

def fig_window(bol= True):
    'open plot inline or qt'
    windows_mode = (
        'qt' if bol==True   # separate window
        else 'inline'   )   # console
    get_ipython().run_line_magic('matplotlib', windows_mode)


class DiffusionWithConsumption3D:
    def __init__(self, params, n0_func, c_func):
        'c_func: needs to be normalized to 1'
        # Physical System Constants
        self.R = params['R']        # Radius of the Diatom (inner radius)
        self.L = params['L']        # Radius of the source (outer radius)
        self.l = self.L - self.R    # Length of the domain
        self.T = params['T']        # Simulation time
        self.Tc = params['Tc']      # Consumption time scale
        self.Td = params['Td']      # Diffusion time scale
        self.Nb = params['Nb']      # Total num of bacteria
        # Diffusion coefficient
        self.D = self.l**2 / (2 * self.Td)
        # Rate of consumption
        self.alpha = 1.0 / self.Tc
        
        # Equation String
        self.pde_str = '∂n/∂t = D/r² ∂/∂r[r² ∂n/∂r] − α n(r) c(r)'
        
        # Num of point for Space/Time grids
        self.nr = params['nr']    # Num radial points
        self.nt = params['nt']    # Num time points
        
        # Space/Time Discretisation
        self.dr = self.l / (self.nr - 1)
        self.r = np.linspace(self.R, self.L, self.nr)
        self.t = np.linspace(0, self.T, self.nt)
        
        # Initial and boundary conditions
        self.n0 = n0_func(self.r)               # Nutrients initial condition
        self.c_dist = self.Nb * c_func(self.r)  # Bacterial distribution
        self.c_tag = c_func.__name__.split('_')[1]

    def solve_ode(self):
        'Solve the steady-state ODE using finite difference method in 3D'
        # Set up the matrix A and vector b
        A = np.zeros((self.nr, self.nr))
        b = np.zeros(self.nr)
        D_dr2 = self.D / self.dr**2
        
        # Fill the tridiagonal matrix A and the vector b
        for i in range(1, self.nr - 1):
            r_i   = self.r[i]
            r_ip1 = self.r[i + 1]
            r_im1 = self.r[i - 1]
            
            # Diagonal for n_(i-1)
            A[i, i-1] = D_dr2 * ( r_im1 / r_i )**2
            # Diagonal for n_i
            A[i, i] = - D_dr2 * (r_ip1**2 + r_im1**2)/r_i**2 - self.alpha * self.c_dist[i]
            # Diagonal for n_(i+1)
            A[i, i+1] = D_dr2 * (r_ip1 / r_i)**2
        
        # Boundary conditions
        A[0, 0] = 1
        b[0] = 0    # n(R) = 0
        A[-1, -1] = 1
        b[-1] = 1   # n(L) = 1
        
        # Solve the linear system (A·n=b)
        self.n_steady = np.linalg.solve(A, b)
        
        # Calculate the steady-state flux
        self.flux_steady = -self.D * np.gradient(self.n_steady, self.dr)
        self.abs_flux_steady = np.abs(self.flux_steady)

    def solve_an_4_cconst(self):
        'Solve the steady-state ODE analytically using sympy'
        # Define the symbols and the function
        r = sp.symbols('r', real=True, positive=True)
        n = sp.Function('n')(r)
        R, L, D, alpha, c0 = sp.symbols('R L D alpha c0', real=True, positive=True)
        
        # Define the ODE '∂n/∂t = 0 = D/r² ∂/∂r[r² ∂n/∂r] − α n(r) c0'
        # ode = sp.Eq(D/r**2 * sp.diff(r**2 * sp.diff(n, r), r) - alpha * n * c0, 0)
        ode = sp.Eq(D/r**2 * sp.diff(r**2 * sp.diff(n, r), r), 0)
        
        # Solve the ODE
        general_sol = sp.dsolve(ode, n)
        
        # Apply the boundary conditions
        C1, C2 = sp.symbols('C1 C2')
        boundary_conditions = [
            general_sol.subs({r: R, n: 0}),  # n(r=R)= 0
            general_sol.subs({r: L, n: 1})   # n(r=L)= 1
        ]
        constants_sol = sp.solve(boundary_conditions, (C1, C2))

        # Substitute the constants back into the general solution
        particular_sol = general_sol.subs(constants_sol)
        particular_sol = sp.simplify(particular_sol)
        self.n_eq = particular_sol
        
        # Substitute the specific values for the constants
        specific_values = {R: self.R, L: self.L, D: self.D, alpha: self.alpha, c0: self.Nb}
        specific_sol = particular_sol.subs(specific_values)
        specific_sol = sp.simplify(specific_sol)
        self.n_speq = specific_sol
        
        'DEBUG'
        sp.pprint(particular_sol)
        sp.pprint(specific_sol)
        ''
        
        # Get n(r) as a function and apply it to the r array
        n_specific_sol = specific_sol.rhs
        n_func = sp.lambdify(r, n_specific_sol, 'numpy')
        self.n_an = n_func(self.r)
        
        
        'Flux'
        # Calculate the flux phi
        flux = -D * sp.diff(particular_sol.rhs, r)
        
        # Substitute the specific values for the constants
        specific_flux = flux.subs(specific_values)
        specific_flux = sp.simplify(specific_flux)
        
        # Get the flux as a function and apply it to the r array
        flux_func = sp.lambdify(r, specific_flux, 'numpy')
        self.phi = flux_func(self.r)
        
        # Calculate the absolute flux phi
        abs_flux = sp.Abs(flux)
        
        
        
    def plot(self, filename= None):
        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        'PLOT 1 - NUTRIENT & BACTERIA CONCENTRATION'
        # Plot n(x) steady-state (ODE computationally) if it has been computed
        if hasattr(self, 'n_steady'):
            ax1.plot(self.r, self.n_steady, 'g', label='$n_{Steady}(x)$')
        
        # Plot n(x) steady-state (ODE analytically) if it has been computed
        if hasattr(self, 'n_an'):
            ax1.plot(self.r, self.n_an, 'm', label='$n_{Analytical}(x)$')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('Nutrient Concentration n(x,t)')
        ax1.set_title('Nutrient Diffusion Given a Bacteria Distribution')
        ax1.grid(True)
        
        # Create a twin y-axis for the Bacteria Concentration
        ax1_b = ax1.twinx()
        ax1_b.plot(self.r, self.c_dist, 'r--', label='$c_{'+self.c_tag+'}$(x)')
        ax1_b.set_ylabel('Bacteria Concentration c(x)', color='red')
        ax1_b.tick_params(axis='y', labelcolor='red')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_b.get_legend_handles_labels()
        ax1.legend(lines + [lines2[0]], labels + [labels2[0]], loc='best')

        'PLOT 2 - NUTRIENT FLUX'
        # Plot the steady-state absolute flux
        if hasattr(self, 'abs_flux_steady'):
            ax2.plot(self.r, self.abs_flux_steady, 'g--', label='$|\Phi_{Steady}(x)|$')
            # Last value plot
            diatom_flux_steady = '$|\\Phi_{Steady}(x=0)|$ =' + f'{self.abs_flux_steady[0]:.6f}'
            ax2.plot(self.r[0], self.abs_flux_steady[0], 'ro', label= diatom_flux_steady)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Flux $|\Phi(x)|$')
        ax2.legend()
        ax2.set_title('Absolute Nutrient Flux')
        ax2.grid(True)
        
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)  # Adjust space between subplots

        # Save the figure
        if filename != None:
            plt.savefig(filename, dpi = 500)

        # Show plot
        plt.show()



# def main():
fig_window(0)

syst_params = {
'R' : 1.0,  'L' : 100,  # Spatial Domain
'T' : 1.0, # Time Domain
'Tc': 1.0,  'Td': 1.0, # Consump & Diff times
'Nb': 0.0,             # Total Number of Bacteria
'nr': 50,   'nt': 100  # Num Spatial/Temporal points
}
l = syst_params['L'] - syst_params['R']

'Initial condition'
def n0_linear(r):
    return r / l

def c_const(r):
    return np.ones_like(r)

diffusion_system = DiffusionWithConsumption3D(syst_params, n0_linear, c_const)

diffusion_system.solve_ode()
diffusion_system.solve_an_4_cconst()
diffusion_system.plot()

diffusion_system.n_eq

# if __name__ == '__main__':
# 	main()