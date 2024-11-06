# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:18:28 2024

@author: Jorge Pottiez López-Jurado
"""

import sympy as sp

r = sp.symbols('r', real=True, positive=True)
n = sp.Function('n')(r)
R, L, D, alpha, c0 = sp.symbols('R L D alpha c0', real=True, positive=True)

# Define the ODE '∂n/∂t = 0 = D/r² ∂/∂r[r² ∂n/∂r] − α n(r) c0'
ode = sp.Eq(D/r**2 * sp.diff(r**2 * sp.diff(n, r), r) - alpha * n * c0, 0)
#ode = sp.Eq(D/r**2 * sp.diff(r**2 * sp.diff(n, r), r), 0)
        
# Ensure that the derivative operations are correctly applied, and simplify where necessary
general_sol = sp.dsolve(ode, n)

# Apply the boundary conditions
C1, C2 = sp.symbols('C1 C2')
boundary_conditions = [
    general_sol.subs({r: R, n: 0}),  # n(x=0)= 0
    general_sol.subs({r: L, n: 1})   # n(x=L)= 1
]
constants_sol = sp.solve(boundary_conditions, (C1, C2))



# Substitute the constants back into the general solution
particular_sol = general_sol.subs(constants_sol)
particular_sol = sp.simplify(particular_sol)
n_eq = particular_sol

# Substitute the specific values for the constants
specific_values = {R: 0.1, L: 1, D: 1/2, alpha: 1, c0: 1}
specific_sol = particular_sol.subs(specific_values)
specific_sol = sp.simplify(specific_sol)

# Get n(x) as a function and apply it to the x array
n_specific_sol = specific_sol.rhs
n_func = sp.lambdify(r, n_specific_sol, 'numpy')

#self.n_an = n_func(self.r)

'DEBUG'
sp.pprint(general_sol)
sp.pprint(particular_sol)
sp.pprint(specific_sol)

# Simplify the ODE expression after substituting the solution back in
check_ode = sp.simplify(ode.subs({n: particular_sol.rhs}))

sp.pprint(check_ode)