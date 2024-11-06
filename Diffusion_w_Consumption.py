# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:29:51 2024

@author: Jorge Pottiez López-Jurado
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from IPython import get_ipython
    
def fig_window(bol= True):
    'open plot inline or qt'
    windows_mode = (
        'qt' if bol==True   # separate window
        else 'inline'   )   # console
    get_ipython().run_line_magic('matplotlib', windows_mode)

def configure_rc_params():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5
    })

class DiffusionWithConsumption1D:
    def __init__(self, params, n0_func, c_func):
        'c_func: needs to be normalized to 1'
        # Physical Syst Constants
        self.L = params['L']      # Length of the domain
        self.T = params['T']      # Simulation time
        self.Tc = params['Tc']    # Consumption time scale
        self.Td = params['Td']    # Diffusion time scale
        self.Nb = params['Nb']    # Total num of bacteria
        # Diffusion coefficient
        self.D = self.L**2 / (2 * self.Td)
        # Rate of consumption
        self.alpha = 1.0 / self.Tc
        
        # Equation String
        self.pde_str = '∂n/∂t = D ∂²n/∂x² − α n(x) c(x)'
        
        # Num of point for Space/Time grids
        self.nx = params['nx']    # Num spatial points
        self.nt = params['nt']    # Num time points
        
        # Space/Time Discretisation
        self.dx = self.L / (self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)
        self.t = np.linspace(0, self.T, self.nt)
        
        # Concentrations Distribution
        self.n0 = n0_func(self.x)           # Nutrients initial cond (PDE)
        self.c_dist = self.Nb*c_func(self.x)    # Bacterial distribution
        self.c_tag = c_func.__name__.split('_')[1]
        
        

    def rhs_pde(self, t, n):
        'represents the right-hand side of the PDE using the finite difference method'
        dn_dt = np.zeros_like(n)
        
        # Boundary conditions (redundant due to initialization)
        dn_dt[0]  = 0
        dn_dt[-1] = 0
        
        # Apply the finite difference method
        for i in range(1, self.nx - 1):
            d2n_dx2 = (n[i + 1] - 2 * n[i] + n[i - 1]) / self.dx**2
            dn_dt[i] = self.D * d2n_dx2 - self.alpha * n[i] * self.c_dist[i]
        
        return dn_dt

    def solve_pde(self):
        'Solve the PDE using Runge-Kutta method of order 5(4)'
        solution = solve_ivp(
            self.rhs_pde, [0, self.T], self.n0, t_eval=self.t, method='RK45'
        )
        # Nutrient Concentration Solution
        self.n = solution.y.T
        
        # Calculate the absolute flux
        self.flux = [ - self.D * np.gradient(n_i, self.dx) for n_i in self.n]
        self.abs_flux = np.abs(self.flux)
        
        # Extract the absolute flux at x=0 for each time point
        self.abs_flux_at_x0 = [flux[0] for flux in self.abs_flux]


    def solve_ode(self):
        'Solve the steady-state ODE using finite difference method'
        # Set up the matrix A and vector b
        A = np.zeros((self.nx, self.nx))
        b = np.zeros(self.nx)
        
        # Fill the tridiagonal matrix A and the vector b
        for i in range(1, self.nx - 1):
            # Diag for n_(i-1)
            A[i, i - 1] = self.D / self.dx**2
            # Diag for n_i
            A[i, i] = -2 * self.D / self.dx**2 - self.alpha*self.c_dist[i]
            # Diag for n_(i+1)
            A[i, i + 1] = self.D / self.dx**2
        
        # Boundary conditions
        A[0, 0] = 1
        b[0] = 0  # n(0) = 0
        A[-1, -1] = 1
        b[-1] = 1  # n(L) = 1
        
        # Solve the linear system (A·n=b)
        self.n_steady = np.linalg.solve(A, b)
        
        # Calculate the steady-state flux
        self.flux_steady = -self.D * np.gradient(self.n_steady, self.dx)
        self.abs_flux_steady = np.abs(self.flux_steady)

    def solve_an_4_cconst(self):
        'Solve the steady-state ODE analytically using sympy'
        # Define the symbols and the function
        x = sp.symbols('x')
        n = sp.Function('n')(x)
        L, D, alpha, c0 = sp.symbols('L D alpha c0', positive=True, real=True)

        # Define the ODE '∂n/∂t = 0 = D ∂²n/∂x² − α n(x) c0'
        ode = sp.Eq(D * sp.diff(n, x, x) - alpha * n * c0, 0)

        # Solve the ODE
        general_sol = sp.dsolve(ode, n)
        
        # Apply the boundary conditions
        C1, C2 = sp.symbols('C1 C2')
        boundary_conditions = [
            general_sol.subs({x: 0, n: 0}),  # n(x=0)= 0
            general_sol.subs({x: L, n: 1})   # n(x=L)= 1
        ]
        constants_sol = sp.solve(boundary_conditions, (C1, C2))

        # Substitute the constants back into the general solution
        particular_sol = general_sol.subs(constants_sol)
        particular_sol = sp.simplify(particular_sol)
        self.n_eq = particular_sol
        
        # Substitute the specific values for the constants
        specific_values = {L: self.L, D: self.D, alpha: self.alpha, c0: self.Nb}
        specific_sol = particular_sol.subs(specific_values)
        specific_sol = sp.simplify(specific_sol)
        self.n_speq = specific_sol
        
        # Get n(x) as a function and apply it to the x array
        n_specific_sol = specific_sol.rhs
        n_func = sp.lambdify(x, n_specific_sol, 'numpy')
        self.n_an = n_func(self.x)


    def plot(self, filename= None):
        # Create the figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        'PLOT 1 - NUTRIENT & BACTERIA CONCENTRATION'
        # Plot n(x,t) over time (PDE computationally)
        colors = viridis(np.linspace(0, 1, self.nt // 10))
        for idx, i in enumerate(range(0, self.nt, self.nt // 10 + 1)):
            # '//' operator is used for floor division
            ax1.plot(self.x, self.n[i], label=f'n(x, t={self.t[i]:.2f})', color=colors[idx])
        
        # Plot n(x) steady-state (ODE computationally) if it has been computed
        if hasattr(self, 'n_steady'):
            ax1.plot(self.x, self.n_steady, 'g--', label='$n_{Steady}(x)$')
        
        # Plot n(x) steady-state (ODE analytically) if it has been computed
        if hasattr(self, 'n_an'):
            ax1.plot(self.x, self.n_an, 'm:', label='$n_{Analytical}(x)$')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('Nutrient Concentration n(x,t)')
        ax1.set_title('Nutrient Diffusion Given a Bacteria Distribution')
        ax1.grid(True)
        
        # Create a twin y-axis for the Bacteria Concentration
        ax1_b = ax1.twinx()
        ax1_b.plot(self.x, self.c_dist, 'r--', label='$c_{'+self.c_tag+'}$(x)')
        ax1_b.set_ylabel('Bacteria Concentration c(x)', color='red')
        ax1_b.tick_params(axis='y', labelcolor='red')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_b.get_legend_handles_labels()
        ax1.legend(lines + [lines2[0]], labels + [labels2[0]], loc='best')


        'PLOT 2 - NUTRIENT FLUX'
        # Plot the Flux over time
        for idx, i in enumerate(range(0, self.nt, self.nt // 10 + 1)):
            # '//' operator is used for floor division
            ax2.plot(self.x, self.abs_flux[i], label=f'$|\Phi(x, t={self.t[i]:.2f})|$', color=colors[idx])
        
        # Plot the steady-state absolute flux if it has been computed
        if hasattr(self, 'abs_flux_steady'):
            ax2.plot(self.x, self.abs_flux_steady, 'g--', label='$|\Phi_{Steady}(x)|$')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Flux $|\Phi(x)|$')
        ax2.legend()
        ax2.set_title('Absolute Nutrient Flux')
        ax2.grid(True)
        
        'PLOT 3 - FLUX @ x=0'
        # Plot the evolution of the Absolute Flux at x=0
        ax3.plot(self.t, self.abs_flux_at_x0, 'b-', label='$|\Phi(x=0, t)|$')
        # Flux's last value
        diatom_flux = f'$|\\Phi(x=0, t={self.t[-1]})|$ = {self.abs_flux[-1][0]:.6f}'
        print(diatom_flux.replace('$',''))
        # Last value plot
        ax3.plot(self.t[-1], self.abs_flux_at_x0[-1], 'ro', label= diatom_flux)
        
        # Plot the steady-state absolute flux at x=0 if it has been computed
        if hasattr(self, 'abs_flux_steady'):
            diatom_flux_steady = '$|\\Phi_{Steady}(x=0)|$ =' + f'{self.abs_flux_steady[0]:.6f}'
            ax3.axhline(self.abs_flux_steady[0], color='g', linestyle='--', label= diatom_flux_steady)
        
        ax3.set_xlabel('t')
        ax3.set_ylabel('$|\Phi(x=0, t)|$')
        ax3.legend()
        ax3.set_title('Absolute Flux evolution for the Diatom')
        ax3.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)  # Adjust space between subplots

        # Save the figure
        if filename != None:
            plt.savefig(filename, dpi = 500)

        # Show plot
        plt.show()


def main():
    fig_window(True)
    
    syst_params = {
    'L' : 1.0,  'T' : 1.0, # Lenght & Time Domain
    'Tc': 1.0,  'Td': 1.0, # Consump & Diff times
    'Nb': 1.0,             # Total Number of Bacteria
    'nx': 50,   'nt': 100  # Num Spatial/Temporal points
    }
    L = syst_params['L']
    
    'Initial condition'
    def n0_linear(x):
        return x / L
    
    def c_const(x):
        return np.ones_like(x)
    
    def c_midstep(x):
        x0 = L/3 # Starting point of the step
        l  = L/3 # Length of the step
        cond = (x >= x0) & (x <= x0 + l)
        return np.where( cond , 1/l , 0)
    
    diffusion_system = DiffusionWithConsumption1D(syst_params, n0_linear, c_midstep)
    diffusion_system.solve_pde()
    diffusion_system.solve_ode()
    diffusion_system.solve_an_4_cconst()
    diffusion_system.plot()

if __name__ == '__main__':
	main()