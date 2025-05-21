import sympy as sp
from scipy.linalg import solve_banded

from base import DiffusionPlotter, np

class Solver3D:
    def __init__(self, params, c_func):
        """
        Solves the 3D Diffusion with Consumption problem using the Finite Difference Method.
        
        Args:
            params (dict): System parameters.
            c_func (callable): Bacterial distribution function, normalized to 1.
        """
        # System Constants
        self.R_dtm  = params['R_dtm']   # Diatom Radius
        self.R_inf  = params['R_inf']   # External Radius
        self.l  = self.R_inf - self.R_dtm   # Domain Length
        self.Tc = params['Tc']      # Consumption Time
        self.alpha = 1.0 / self.Tc  # Consumption Rate
        # Discretisation Constant
        self.nr = params['nr']
        # Shell Parameters (if they exist)
        self.rho = params.get('rho', None)
        self.lambda_ = params.get('lambda', None)

        self.__discretise_system(c_func)

        self.ode = self.SolveODE(self)
        self.pde = None     # Implementation of the PDE Solver is not necessary
        self.plot = DiffusionPlotter(self, x_str='r')
    
    def __discretise_system(self, c_func):
        """Discretise Space and input distributions."""
        # Space Discretisation
        self.dr = self.l / (self.nr - 1)
        self.r = np.linspace(self.R_dtm, self.R_inf, self.nr)
        
        # Bacterial Distribution
        self.c = c_func(self.r)
        self.c_tag = c_func.__name__.split('_')[1]

    class SolveODE:
        def __init__(self, parent):
            self.parent = parent
            self.eq_str = '1/r² ∂/∂r[r² ∂n/∂r] − α n(r) c(r) = 0'
            self.analyt = self.SolveAnalytically(self)
        
        def solve(self, optimised=True):
            if optimised:
                self.__solve_opt()  # O(nr) complexity
            else:
                self.__solve()      # O(nr³) complexity
            self.__calculate_flux()
        
        def __solve(self):
            """
            Solve the steady-state ODE using finite difference method in 3D.
            Using the standard numpy.linalg.solve method.
            """
            syst = self.parent # Parent's alias
            A = np.zeros((syst.nr, syst.nr))
            b = np.zeros(syst.nr)
            _dr2 = 1 / syst.dr**2
            _dr  = 1 / syst.dr
            # Fill the tridiagonal matrix A and the vector b
            for i in range(1, syst.nr - 1):
                A[i, i-1] = _dr2 - _dr / syst.r[i]
                A[i, i] = - 2*_dr2 - syst.alpha * syst.c[i]
                A[i, i+1] = _dr2 + _dr / syst.r[i]
            # Boundary conditions
            A[0, 0] = A[-1, -1] = 1.0
            b[0]  = 0.0       # n(r=R_dtm) = 0
            b[-1] = 1.0       # n(r=R_inf) = 1
            # Solve A·n=b
            self.n = np.linalg.solve(A, b)

        def __solve_opt(self):
            """
            Solve the steady-state ODE using finite difference method in 3D.
            Optimised version using scipy.linalg.solve_banded.
            """
            syst = self.parent
            ab = np.zeros((3, syst.nr))  # 'ab' will hold [superdiag, diag, subdiag]
            b = np.zeros(syst.nr)
            
            # Boundary conditions
            # n(r=R_dtm) = 0 --> row 0 enforces n[0] = 0
            ab[1, 0] = 1.0
            b[0] = 0.0

            # n(r=R_inf) = 1 --> row -1 enforces n[-1] = 1
            ab[1, -1] = 1.0
            b[-1] = 1.0

            _dr2 = 1 / syst.dr**2
            _dr  = 1 / syst.dr
            
            i_idx = np.arange(1, syst.nr - 1)
            # main diagonal
            ab[1, i_idx]     = -2.0*_dr2 - syst.alpha * syst.c[i_idx]
            # subdiagonal (stored in ab[2, i_idx-1])
            ab[2, i_idx - 1] = _dr2 - _dr / syst.r[i_idx]
            # superdiagonal (stored in ab[0, i_idx+1])
            ab[0, i_idx + 1] = _dr2 + _dr / syst.r[i_idx]

            # Solve A·n = b
            self.n = solve_banded((1, 1), ab, b) # (1,1): 1 superdiagonal and 1 subdiagonal
            
        def __calculate_flux(self):
            """Calculate the steady-state flux."""
            self.flux = - np.gradient(self.n, self.parent.dr)
            self.abs_flux = np.abs(self.flux)
        

        class SolveAnalytically:
            def __init__(self, parent):
                self.parent = parent
            
            def solve(self):
                """Solve the steady-state ODE analytically."""
                syst = self.parent.parent # Grandparent's alias
                
                if syst.rho is not None and syst.lambda_ is not None:
                    # compute self.n
                    self.__solve_c_shell_analytically(syst.r)
                else:
                    raise ValueError("Shell parameters (rho, lambda) are not defined.")
                
                # Calculate the flux
                self.flux = - np.gradient(self.n, syst.dr)
                self.abs_flux = np.abs(self.flux)

            def __solve_c_shell_analytically(self, r):
                """Solve the steady-state ODE analytically for c_shell(r)"""
                syst = self.parent.parent # Grandparent's alias

                # Grab shell parameters
                # Define parameters
                c0 = 3/(4 * np.pi) * 1 / (syst.lambda_**3 + 3 * syst.lambda_**2 * syst.rho + 3 * syst.lambda_ * syst.rho**2)
                k = np.sqrt(syst.alpha * c0)
                sinh_term = np.sinh(k * syst.lambda_)
                cosh_term = np.cosh(k * syst.lambda_)
                
                # Compute constants
                ### Inner region (r < R_dtm)
                denom_A = k * (syst.rho - syst.R_dtm) * sinh_term + cosh_term
                A = - syst.R_dtm / denom_A
                ### Shell region (R_dtm <= r < r0)
                B = (
                        A * (k * (syst.R_dtm - syst.rho) - 1) / (2 * k * syst.R_dtm)
                    ) * np.exp(-k * syst.rho)
                C = (
                        A * (k * (syst.R_dtm - syst.rho) + 1) / (2 * k * syst.R_dtm)
                    ) * np.exp( k * syst.rho)
                ### Outer region (r > r0)
                numer_D = A * (k * (syst.R_dtm - syst.rho) * np.cosh(k * syst.lambda_) - np.sinh(k * syst.lambda_))
                D = numer_D / (k * syst.R_dtm) - (syst.rho + syst.lambda_)
                
                # Create the piecewise function for n(r)
                result = np.zeros_like(r, dtype=float)
                
                ### Apply each condition and formula
                inner_region = (r >= syst.R_dtm) & (r < syst.rho)
                result = np.where(inner_region, A * (1/r - 1/syst.R_dtm), result)
                
                shell_region = (r >= syst.rho) & (r <= syst.rho + syst.lambda_)
                result = np.where(shell_region, (B * np.exp(k * r) + C * np.exp(-k * r)) / r, result)
                
                outer_region = r > syst.rho + syst.lambda_
                result = np.where(outer_region, D/r + 1, result)

                self.n = result


    ''' Deprecated: Analytic solution using sympy is not necessary and too enoying to implement. '''
        # class SolveAnalytically:
        #     def __init__(self, parent):
        #         self.parent = parent
            
        #     def solve_c_const(self):
        #         self.solve_c_const_analytically()
        #         self.calculate_flux()
            
        #     def solve_c_const_analytically(self):
        #         """Solve the steady-state ODE analytically using sympy."""
        #         # Problem definition
        #         self.r = r = sp.symbols('r', real=True, positive=True)
        #         n = sp.Function('n')(r)
        #         R_dtm, R_inf, alpha = sp.symbols('R_dtm R_inf alpha', real=True, positive=True)
        #         ode = sp.Eq(1/r**2 * sp.diff(r**2 * sp.diff(n, r), r) - alpha * n, 0)
                
        #         # Solve the ODE and apply the boundary conditions
        #         self.general_sol = sp.dsolve(ode, n)
        #         constants_sol = self._apply_boundary_conditions(r, n, R_dtm, R_inf)
        #         self.particular_sol = sp.simplify(
        #             self.general_sol.subs(constants_sol)
        #         )
        #         # Apply the numeric values
        #         self.numeric_sol = self._substitute_numeric_values(R_dtm, R_inf, alpha)
        #         n_func = sp.lambdify(r, self.numeric_sol.rhs, 'numpy')
        #         self.n = n_func(self.parent.parent.r)

        #         # Substitute the solution back into the ODE
        #         # check_consistency = sp.simplify(ode.subs({n: self.particular_sol.rhs}))
        #         # print(check_consistency)
            
        #     def _apply_boundary_conditions(self, r, n, R_dtm, R_inf):
        #         """Apply the boundary conditions."""
        #         C1, C2 = sp.symbols('C1 C2')
        #         boundary_conditions = [
        #             self.general_sol.subs({r: R_dtm, n: 0}),  # n(r=R_dtm)= 0
        #             self.general_sol.subs({r: R_inf, n: 1})   # n(r=R_inf)= 1
        #         ]
        #         return sp.solve(boundary_conditions, (C1, C2))

        #     def _substitute_numeric_values(self, R_dtm, R_inf, alpha):
        #         """Substitute the numeric values for the constants."""
        #         syst = self.parent.parent # Grandparent's alias

        #         self.__numeric_values = {
        #             R_dtm: syst.R_dtm, R_inf: syst.R_inf, alpha: syst.alpha
        #             }
        #         numeric_sol = self.particular_sol.subs(self.__numeric_values)
        #         return sp.simplify(numeric_sol)
            
        #     def calculate_flux(self):
        #         """Calculate the steady-state flux analytically."""
        #         syst = self.parent.parent # Grandparent's alias
        #         phi = sp.Function('Phi')(self.r)
        #         # Flux related to the solution with the boundary Conditions
        #         self.particular_flux = sp.Eq(
        #             phi, - sp.diff(self.particular_sol.rhs, self.r)
        #         )
        #         # Substitute the numeric values for the constants
        #         self.numeric_flux = sp.simplify(
        #             self.particular_flux.subs(self.__numeric_values)
        #         )
        #         # Get the flux as a function and apply it to the r array
        #         flux_func = sp.lambdify(self.r, self.numeric_flux.rhs, 'numpy')
        #         self.flux = flux_func(syst.r)
        #         # Calculate the absolute flux phi
        #         self.abs_flux = np.abs(self.flux)

        #     def prints(self):
        #         """Pretty print the solutions."""
        #         print('General solution:')
        #         sp.pprint(self.general_sol)
        #         print('Particular solution:')
        #         sp.pprint(self.particular_sol)
        #         print('Numerical solution:')
        #         sp.pprint(self.numeric_sol)
            
        #     def prints_flux(self):
        #         """Pretty print the flux solutions."""
        #         print('Particular flux:')
        #         sp.pprint(self.particular_flux)
        #         print('Numerical flux:')
        #         sp.pprint(self.numeric_flux)