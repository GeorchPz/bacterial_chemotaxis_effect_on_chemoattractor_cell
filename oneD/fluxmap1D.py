from base import BaseFluxMap, np
from .solver1D import Solver1D

def c_none (x):
    'Dummy function for no bacterial step distribution'
    return None

class FluxMap1D(BaseFluxMap):
    @property
    def _solver_class(self):
        return Solver1D

    def _init_values(self):
        # ODE parameters
        self.L       = self.params['L']
        self.T_ratio = self.params['T_ratio']
        # Compute Timescale parameters based on the T_ratio
        extract_params = Solver1D(self.params, c_none)
        self.Td      = extract_params.Td
        self.Tc      = extract_params.Tc
        self.alpha   = extract_params.alpha
        
        # FluxMap parameters
        self.n_x = self.params['n_x0']
        self.n_y = self.params['n_l']
        self.x_values = np.linspace(0, self.L, self.n_x)
        self.y_values = np.linspace(0, self.L, self.n_y  + 1)[1:]
    
    def _plot_annotations(self):
        self.xlabel = 'Starting Point ($x_0$)'
        self.ylabel = 'Step Length ($l$)'
        self.title  = (
            'Diatom Flux under Varying Bacterial Step Distributions'
            + f'\n$\\tau_c/\\tau_d={self.T_ratio}$'
        )