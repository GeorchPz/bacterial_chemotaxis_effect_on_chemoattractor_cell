from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded

from base import DiffusionPlotter, np

class Solver1D:
    def __init__(self, params, c_func, n0_func = None):
        """
        Solves the 1D Diffusion with Consumption problem using the Finite Difference Method.
        
        Args:
            params (dict): System parameters.
            c_func (callable): Bacterial distribution function.
            n0_func (callable, optional): Initial condition for the Nutrients Concentration.
        """
        # System Constants
        self.L  = params['L']       # Domain Length
        self.Tc = params['Tc']      # Consumption Time
        self.alpha = 1.0 / self.Tc  # Consumption Rate
        # Discretisation Constant
        self.nx = params['nx']

        if n0_func is not None:
            # PDE related constants
            self.T  = params['T']
            self.nt = params['nt']
        
        self.__discretise_system(n0_func, c_func)
        
        self.ode = self.ODESolver(self)
        if n0_func is not None:
            self.pde = self.PDESolver(self)
        self.plot = DiffusionPlotter(self)

    def __discretise_system(self, n0_func, c_func):
        """Discretise Space/Time and input distributions."""
        self.dx = self.L / (self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)
        self.c = c_func(self.x)
        self.c_tag = c_func.__name__.split('_')[1]

        if n0_func is not None:
            self.t = np.linspace(0, self.T, self.nt)
            self.n0 = n0_func(self.x)

    def __stability_condition(self):
        """Calculate the stability condition for the PDE, Runge-Kutta 5(4)."""
        # Eigenvalue from discretisation
        R = 1.6 # Stability factor for RK5(4)
        lambd = - 2 / self.dx**2 - self.alpha * np.max(self.c)
        dt = R / abs(lambd)
        nt = int(self.T / dt + 1)
        return nt

    class ODESolver:
        def __init__(self, parent):
            self.parent = parent
            self.eq_str = '∂²n/∂x² − α n(x) c(x) = 0'
        
        def solve(self, optimised=True):
            if optimised:
                self.__solve_ode_opt()  # O(nx) complexity
            else:
                self.__solve_ode()      # O(nx³) complexity
            self.__calculate_flux()

        def __solve_ode(self):
            """
            Solve the steady-state ODE using finite difference method.
            Using the standard numpy.linalg.solve method.
            """
            syst = self.parent # Parent's alias
            A = np.zeros((syst.nx, syst.nx))
            b = np.zeros(syst.nx)
            _dx2 = 1 / syst.dx**2
            # Fill the tridiagonal matrix A and the vector b
            for i in range(1, syst.nx - 1):
                A[i, i-1] = _dx2        # = n_(i-1)
                A[i, i] =  -2.0*_dx2 - syst.alpha * syst.c[i] # = n_i
                A[i, i+1] = _dx2        # = n_(i+1)
            # Boundary conditions
            A[0, 0] = A[-1, -1] = 1.0
            b[0]  = 0.0     # n(x=0) = 0
            b[-1] = 1.0     # n(x=L) = 1
            # Solve A·n = b
            self.n = np.linalg.solve(A, b)

        def __solve_ode_opt(self):
            '''
            Solve the steady-state ODE using finite difference method.
            Optimised version of the method using numpy.linalg.solve.
            '''
            syst = self.parent
            ab = np.zeros((3, syst.nx))  # 'ab' will hold [superdiag, diag, subdiag]
            b = np.zeros(syst.nx)

            # Boundary conditions
            ab[1, 0] = ab[1, -1] = 1.0
            b[0]  = 0.0     # n(x=0) = 0
            b[-1] = 1.0     # n(x=L) = 1

            _dx2 = 1 / syst.dx**2

            i_idx = np.arange(1, syst.nx - 1)
            # Fill the diagonals
            ab[1, i_idx] = -2.0 * _dx2 - syst.alpha * syst.c[i_idx]
            ab[2, i_idx - 1] = ab[0, i_idx + 1] = _dx2

            # Solve A·n = b
            self.n = solve_banded((1, 1), ab, b) # (1,1: 1 superdiagonal, 1 subdiagonal)

        def __calculate_flux(self):
            """Calculate the steady-state flux."""
            self.flux = -np.gradient(self.n, self.parent.dx)
            self.abs_flux = np.abs(self.flux)
    
    class PDESolver:
        def __init__(self, parent):
            self.parent = parent
            self.eq_str = '∂n/∂t = ∂²n/∂x² − α n(x) c(x)'
        
        def solve(self):
            self.__solve_pde()
            self.__calculate_flux()

        def __rhs_pde(self, t, n):
            """Right-hand side of the PDE using finite difference method."""
            syst = self.parent # Parent's alias
            dn_dt = np.zeros_like(n)
            # Boundary conditions
            dn_dt[0] = dn_dt[-1] = 0
            
            for i in range(1, syst.nx - 1):
                d2n_dx2 = (n[i + 1] - 2 * n[i] + n[i - 1]) / syst.dx**2
                dn_dt[i] = d2n_dx2 - syst.alpha * n[i] * syst.c[i]
            
            return dn_dt

        def __solve_pde(self):
            """Solve the PDE using Runge-Kutta method of order 5(4)."""
            solution = solve_ivp(
                self.__rhs_pde, [0, self.parent.T], self.parent.n0, t_eval=self.parent.t, method='RK45'
            )
            self.n = solution.y.T

        def __calculate_flux(self):
            """Calculate the absolute flux."""
            self.flux = [-np.gradient(n_i, self.parent.dx) for n_i in self.n]
            self.abs_flux = np.abs(self.flux)
            # Diatomic absolute flux
            self.abs_flux_at_x0 = [flux[0] for flux in self.abs_flux]