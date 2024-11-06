# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 10:57:41 2024

@author: Jorge Pottiez López-Jurado
"""

from sympy import symbols, sin, cos
import numpy as np
import inspect

def antiLambdify(func, args):
    # Extract the function's code as a string
    source = inspect.getsource(func)
    
    # Define symbols for the arguments
    sympy_args = symbols(args)
    
    # Create a mapping of NumPy functions to SymPy equivalents
    func_mapping = {
        'np.sin': 'sin',
        'np.cos': 'cos',
        'np.ones_like(x)': '1',
        'np.ones_like(r)': '1',
        # Add more mappings
    }
    
    # Remove the function definition line and dedent the body
    _, eq_str = source.split('return ')
    
    # Replace NumPy functions with SymPy equivalents
    for np_func, sympy_func in func_mapping.items():
        eq_str = eq_str.replace(np_func, sympy_func)
    
    # Create a local dictionary to execute the code
    local_dict = {str(arg): sympy_arg for arg, sympy_arg in zip(args, sympy_args)}
    
    # Execute the modified source to get the sympy expression
    exec(eq_str, globals(), local_dict)
    
    print(local_dict)
    
    return local_dict['expr']

# Example usage
def numerical_function(x, y):
    return 2 * np.sin(x) + np.cos(y) + np.ones_like(x)

# Convert the numerical function to a SymPy function
args = 'x y'
sympy_expr = antiLambdify(numerical_function, args)
sympy_expr
