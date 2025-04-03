from base import BaseFluxMap, np
from .solver1D import Solver1D

class FluxMap1D(BaseFluxMap):
    @property
    def _solver_class(self):
        return Solver1D

    def _init_values(self):
        # ODE parameters
        self.L  = self.params['L']
        self.Tc = self.params['Tc']
        # FluxMap parameters
        self.n_x = self.params['n_x0']
        self.n_y  = self.params['n_l']
        self.x_values = np.linspace(0, self.L, self.n_x)
        self.y_values  = np.linspace(0, self.L, self.n_y  + 1)[1:]
    
    def _plot_annotations(self):
        self.xlabel = 'Starting Point ($x_0$)'
        self.ylabel = 'Step Length ($l$)'
        self.title = (
            'Diatom Flux under Varying Bacterial Step Distributions'
            + f'\n$T_c={self.Tc}$'
        )