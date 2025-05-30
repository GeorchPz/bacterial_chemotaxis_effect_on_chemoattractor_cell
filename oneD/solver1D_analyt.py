import sympy as sp

from base import DiffusionPlotter, plt, np
from .solver1D import Solver1D

class Solver1D_UniformBacterium(Solver1D):
    def __init__(self, params: dict, initial_condition: callable):
        super().__init__(params, self.c_const, initial_condition)
        self.ode = self.ODESolver(self)

    @staticmethod
    def c_const(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    class ODESolver(Solver1D.ODESolver):
        def __init__(self, parent):
            super().__init__(parent)
            self.analyt = self.AnalyticalSolver(self)

        class AnalyticalSolver:
            def __init__(self, parent):
                self.parent = parent

            def solve(self):
                """Solve the steady-state ODE analytically using sympy."""
                # Problem definition
                x = sp.symbols('x', real=True)
                n = sp.Function('n')(x)
                L, alpha = sp.symbols('L alpha', real=True, positive=True)
                ode = sp.Eq(sp.diff(n, x, x) - alpha * n, 0)

                # Solve the ODE and apply the boundary conditions
                self.general_sol = sp.dsolve(ode, n)
                constants_sol = self._apply_boundary_conditions(x, n, L)
                self.particular_sol = sp.simplify(
                    self.general_sol.subs(constants_sol)
                    )
                # Apply the numeric values
                self.numeric_sol = self._substitute_numeric_values(L, alpha)
                self.n_func = sp.lambdify(x, self.numeric_sol.rhs, 'numpy')
                self.n = self.n_func(self.parent.parent.x)

            def _apply_boundary_conditions(self, x, n, L):
                """Apply the boundary conditions."""
                C1, C2 = sp.symbols('C1 C2')
                boundary_conditions = [
                    self.general_sol.subs({x: 0, n: 0}),
                    self.general_sol.subs({x: L, n: 1})
                ]
                return sp.solve(boundary_conditions, (C1, C2))

            def _substitute_numeric_values(self, L, alpha):
                """Substitute the specific values for the constants."""
                system = self.parent.parent # Grandparent's alias

                numeric_values = {
                    L: system.L, alpha: system.alpha
                }
                numeric_sol = self.particular_sol.subs(numeric_values)
                return sp.simplify(numeric_sol)
            
            def print_solutions(self):
                """Pretty print the solutions."""
                print('General solution:')
                sp.pprint(self.general_sol)
                print('Particular solution:')
                sp.pprint(self.particular_sol)
                print('Numeric solution:')
                sp.pprint(self.numeric_sol)