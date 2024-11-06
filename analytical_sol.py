# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 10:46:09 2024

@author: Jorge Pottiez López-Jurado
"""

import sympy as sp

# Define the symbols and the function
x = sp.symbols('x')
n = sp.Function('n')(x)
D, alpha, c0 = sp.symbols('D alpha c0', positive=True, real=True)

# Define the ODE '∂n/∂t = 0 = D ∂²n/∂x² − α n(x) c(n)'
ode = sp.Eq(D * sp.diff(n, x, x) - alpha * n * c0, 0)

# Solve the ODE
general_solution = sp.dsolve(ode, n)
n_sol = general_solution.rhs

# Apply the boundary conditions
C1, C2 = sp.symbols('C1 C2')
boundary_conditions = [
    sp.Eq(n_sol.subs(x, 0), 0), # n(x=0)= 0
    sp.Eq(n_sol.subs(x, 1), 1), # n(x=L)= 1
    ]
constants_solution = sp.solve(boundary_conditions, (C1, C2))

# Substitute the constants back into the general solution
n_specific_solution = n_sol.subs(constants_solution)

print('n(x)=')
sp.pprint(n_specific_solution)



# Define the specific values for D, alpha, c0
vals_dict = {D: 0.5, alpha: 1, c0: 1}

# Substitute the specific values into the solution
n_specific_solution = n_specific_solution.subs(vals_dict)

# Convert the specific solution to a function
n_function = sp.lambdify(x, n_specific_solution, 'numpy')

# Print the specific solution
print("\nSpecific solution for n(x):")
sp.pprint(n_specific_solution)

# Example: Evaluate the solution at x = 0.5
x_val = 0.5
n_val = n_function(x_val)
print(f"\nn({x_val}) = {n_val}")
