from abc import ABC, abstractmethod     # Define abstract base classes
from sys import stdout                  # For printing progress bar
from tqdm import tqdm                   # For displaying progress bar
from joblib import Parallel, delayed    # For parallel processing

from . import plt, np, h5py
from .base_plotter import BasePlotter


class BaseFluxMap(ABC, BasePlotter):
    def __init__(self, params, c_generator, set_logscale=False):
        self.params = params
        self.c_generator = c_generator
        self.set_logscale = set_logscale

        self._init_values()
        self._plot_annotations()

        self.flux_map = np.full((self.n_x, self.n_y), np.nan)
        self.configure_rc_params()

        self.data_storage_file = 'fluxmaps.h5'

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
            (iter_z, iter_t, z, t)
            for iter_z, z in enumerate(z_values)
            for iter_t, t in enumerate(self.T_ratio_values)
        ]

        def flux_element(iter_z, iter_t, z, T_ratio):
            '''Solves the diffusion system for a single combination of y and T0 values.'''
            c_func = self.c_gen(z)
            self.params['T_ratio'] = T_ratio
            diffusion_system = self._solver_class(self.params, c_func)
            diffusion_system.ode.solve()

            return iter_z, iter_t, diffusion_system.ode.abs_flux[0]

        results = self.__parallel_solve(flux_element, combinations, n_jobs)

        if self.iterate_thick:
            # Permute the results to match the flux axis
            results = [(iter_t, iter_z, phi) for iter_z, iter_t, phi in results]

        # Reconstruction of the flux map
        for i, j, phi in results:
            self.flux_map[i, j] = phi

    def solve(self, n_jobs=None):
        '''Solver method that calls the appropriate solver method based on the number of independent variables.'''
        if type(self.T_ratio) == tuple:
            self.solve_multiple_absorptions(n_jobs)
        else:
            self.solve_single_absorption(n_jobs)
    
    def search_extremes(self, in_ax=False):
        # Find minimum
        min_flux = np.nanmin(self.flux_map)
        min_coords = np.unravel_index(np.nanargmin(self.flux_map), self.flux_map.shape)
        min_x = self.x_values[min_coords[0]]
        min_y = self.y_values[min_coords[1]]
        
        # Find maximum
        max_flux = np.nanmax(self.flux_map)
        max_coords = np.unravel_index(np.nanargmax(self.flux_map), self.flux_map.shape)
        max_x = self.x_values[max_coords[0]]
        max_y = self.y_values[max_coords[1]]
        
        if in_ax:
            # Plot both extremes
            self.ax.plot(
                min_x, min_y, 'bo', markersize= 8, mfc='none',
                label='$|\\phi|_{min}='+f'{min_flux:.3f}$'
                )
            self.ax.plot(
                max_x, max_y, 'ro', markersize= 8, mfc='none',
                label='$|\\phi|_{max}='+f'{max_flux:.3f}$'
                )
        else:
            # Print both extremes
            print(f"Minimum flux: Phi({min_x:.3f}, {min_y:.3f})={min_flux:.3f}")
            print(f"Maximum flux: Phi({max_x:.3f}, {max_y:.3f})={max_flux:.3f}")
            return (min_flux, min_x, min_y), (max_flux, max_x, max_y)
    
    def transition_boundary(self, in_ax=False):
        # Find the positions where these minima occur
        minr0_indices = [np.nanargmin(φi) for φi in self.flux_map.T] # ignoring NaN values
        # Get the corresponding values for the maximum flux
        self.r0_transect = [self.x_values[i] for i in minr0_indices]

        if in_ax:
            # Add the maximum flux line connecting the maxima
            self.ax.plot(self.r0_transect, self.y_values, 'r--', label='$\\text{min}_{\\rho} \\, |\\phi|$', linewidth=1)
    
    def plot(
            self, ax=None, flux_range=(0, 1),
            set_extremes=False, set_transition=False,
            set_xlog=False, set_ylog=False, set_title=False
        ):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 5))
        else:
            self.ax = ax
        
        # Check if extremes needs to be plotted
        self.search_extremes(in_ax=set_extremes)
        # Check if transition boundary needs to be plotted
        if set_transition:
            self.transition_boundary(in_ax=True)

        if set_extremes or set_transition:
            self.ax.legend()

        if set_xlog:
            self.ax.set_xscale('log')
        if set_ylog:
            self.ax.set_yscale('log')

        # Set min and max values for the colormap
        vmin, vmax = flux_range if flux_range else (None, None)
        contour = self.ax.contourf(
            self.x_values, self.y_values,
            self.flux_map.T, 20, cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(contour, ax=self.ax, label=self.cbar_label)
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.grid()

        if set_title:
            self.ax.set_title(self.title)

    def save_data(self, group_name, **kwargs):
        '''Save the data to a file with the given path and filename.'''
        mode='a' # Opens the file for reading and writing, creating it if it doesn’t exist.
        with h5py.File(self.data_storage_file, mode) as f:
            if group_name in f: # Remove the group before overwriting
                del f[group_name]

            group = f.create_group(group_name)
            
            # Save flux map data
            group.create_dataset('xlabel', data=self.xlabel)
            group.create_dataset('ylabel', data=self.ylabel)
            group.create_dataset('xvalues', data=self.x_values)
            group.create_dataset('yvalues', data=self.y_values)
            group.create_dataset('flux_map', data=self.flux_map)
    
    def load_data(self, group_name):
        '''Load the data from a file with the given path and filename.'''
        with h5py.File(self.data_storage_file, mode='r') as f:
            group = f[group_name]
            
            # Load flux map data
            self.flux_map = group['flux_map'][()]