from abc import ABC, abstractmethod     # Define abstract base classes
from sys import stdout                  # For printing progress bar
from tqdm import tqdm                   # For displaying progress bar
from joblib import Parallel, delayed    # For parallel processing

from . import plt
from . import np
from .base_plotter import BasePlotter


class BaseFluxMap(ABC, BasePlotter):
    def __init__(self, params, c_generator, set_logscale=False):
        self.params = params
        self.c_generator = c_generator
        self.set_logscale = set_logscale

        self._init_values()
        self._plot_annotations()

        self.flux_map  = np.zeros((self.n_x, self.n_y))
        self.configure_rc_params()

    @abstractmethod
    def _init_values(self):
        """
        Subclasses must define:
            - self.n_x
            - self.n_y
            - self.x_values
            - self.y_values
        Where x & y are the independent variables of the flux map.
        For example, in 3D, x=r0 and y=y, the starting radius and layer length respectively.
        """
        pass

    @abstractmethod
    def _plot_annotations(self):
        """Subclasses must define the xlabel, ylabel, and title."""
        pass

    @property
    @abstractmethod
    def _solver_class(self):
        """Subclasses must define the solver class to use."""
        pass

    # Function not used, but left for reference purposes,
    # of how the flux map was solved before parallel processing was implemented.
    def solve_not_opt(self):
        '''Solves the diffusion system for each combination of x and y values.'''
        total_iterations = self.n_x * self.n_y
        divisor = 2 * np.sqrt(total_iterations)
        iteration = 0

        for i, x in enumerate(self.x_values):
            for j, y in enumerate(self.y_values):
                if x + y <= self.L:
                    c_func = self.c_generator(x, y)
                    diffusion_system = self._solver_class(self.params, c_func)
                    diffusion_system.ode.solve()
                    self.flux_map[i, j] = diffusion_system.ode.abs_flux[0]

                iteration += 1
                if iteration % divisor == 0 or iteration == total_iterations:
                    percent = iteration / total_iterations * 100
                    stdout.write(f"\rODEs solved: {percent:.2f}%")
                    stdout.flush()

    @staticmethod
    def my_logspace(start, stop, num):
        '''Generates a logarithmic space of num points between start and stop.'''
        return np.logspace(np.log10(start), np.log10(stop), num)

    @staticmethod
    def __parallel_solve(parallel_element, iterations, n_jobs):
        '''
        Parallelises the solving of multiple independent ODEs using joblib.
        To display a progress bar, the tqdm function is used.
        Args:
            parallel_element: function to parallelise
            iterations: list of tuples with the arguments for the parallel_element function
            n_jobs: number of parallel jobs to run
        '''
        iter_num = len(iterations)
        results = Parallel(n_jobs=n_jobs)(
            delayed(parallel_element)(*comb)
            for comb in tqdm(
                iterations,
                desc="ODEs solved",
                total=iter_num
            )
        )
        return results
        
    def solve_single_absorption(self, n_jobs=None):
        '''Solves the diffusion system for each combination of x and y values using parallel processing.'''
        valid_combinations = [
            (i, j, x, y)
            for i, x in enumerate(self.x_values)
            for j, y  in enumerate(self.y_values)
            if x + y <= self.L
        ]
        def flux_element(i, j, x, y):
            '''Solves the diffusion system for a single combination of x and y values.'''
            c_func = self.c_generator(x, y)
            diffusion_system = self._solver_class(self.params, c_func)
            diffusion_system.ode.solve()

            return i, j, diffusion_system.ode.abs_flux[0]
        
        results = self.__parallel_solve(flux_element, valid_combinations, n_jobs)
        # Reconstruction of the flux map
        for i, j, phi in results:
            self.flux_map[i, j] = phi


    def solve_multiple_absorptions(self, n_jobs=None):
        '''Solves the diffusion system for each combination of y and T0 values using parallel processing.'''
        if self.iterate_radii:
            z_values = self.x_values
            self.c_gen = lambda x: self.c_generator(x, self.w)
        elif self.iterate_thick:
            z_values = self.y_values
            self.c_gen = lambda y: self.c_generator(self.w, y)

        combinations = [
            (k, t, z, T)
            for k, z in enumerate(z_values)
            for t, T in enumerate(self.Tc_values)
        ]

        def flux_element(k, t, z, Tc):
            '''Solves the diffusion system for a single combination of y and T0 values.'''
            c_func = self.c_gen(z)
            self.params['Tc'] = Tc
            diffusion_system = self._solver_class(self.params, c_func)
            diffusion_system.ode.solve()

            return k, t, diffusion_system.ode.abs_flux[0]

        results = self.__parallel_solve(flux_element, combinations, n_jobs)

        if self.iterate_thick:
            # Permute the results to match the flux axis
            results = [(t, k, phi) for k, t, phi in results]

        # Reconstruction of the flux map
        for i, j, phi in results:
            self.flux_map[i, j] = phi

    def solve(self, n_jobs=None):
        '''Solver method that calls the appropriate solver method based on the number of independent variables.'''
        if type(self.Tc) == tuple:
            self.solve_multiple_absorptions(n_jobs)
        else:
            self.solve_single_absorption(n_jobs)

    def search_minimum(self, set_ax = True):
        flux_m = np.nanmin(self.flux_map)
        coords_m = np.unravel_index(np.nanargmin(self.flux_map), self.flux_map.shape)
        x_m = self.x_values[coords_m[0]]
        y_m  = self.y_values[coords_m[1]]
        if set_ax:
            self.ax.plot(x_m, y_m, 'ro', label='$|\\phi|_{min}='+f'{flux_m:.3f}$')
            self.ax.legend()
        else:
            print(f"Minimum flux: Phi({x_m:.3f}, {y_m:.3f})={flux_m:.3f}")
            return flux_m, x_m, y_m
        
    def plot(
            self, ax=None, set_nans=False, set_min= False,
            set_xlog=False, set_ylog=False, set_title=True
        ):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        else:
            self.ax = ax

        if set_nans:
            self.flux_map[self.flux_map == 0] = np.nan
        if set_min:
            self.search_minimum()
        if set_xlog:
            self.ax.set_xscale('log')
        if set_ylog:
            self.ax.set_yscale('log')

        contour = self.ax.contourf(
            self.x_values, self.y_values,
            self.flux_map.T, 20, cmap='viridis'
        )
        cbar = plt.colorbar(contour, ax=self.ax, label='$|\\Phi(r=R_{Diatom})|$')
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.grid()

        if set_title:
            self.ax.set_title(self.title)
