from base import BaseFluxMap, np
from .solver3D import Solver3D

def c_none(x):
    'Dummy function for no bacterial step distribution'
    return None

class FluxMap3D(BaseFluxMap):
    @property
    def _solver_class(self):
        return Solver3D

    def _init_values(self):
        # ODE parameters
        self.R_dtm = self.params['R_dtm']
        self.L     = self.params['L']
        self.alpha    = self.params['alpha']
        # Compute Timescale parameters based on the alpha
        extract_params = Solver3D(self.params, c_none)
        self.Td      = extract_params.Td
        self.Tc      = extract_params.Tc
        self.T_ratio = extract_params.T_ratio

        # FluxMap parameters
        self.n_x = self.params.get('n_rho')
        self.n_y = self.params.get('n_lambda')
        self.n_alpha = self.params.get('n_alpha')
        # params.get() returns None if the key is not found

        # Check if we are iterating over thickness or radii
        self.iterate_thick = 'n_lambda' in self.params
        self.iterate_radii = 'n_rho' in self.params

        if self.n_x:
            self.x_values = np.linspace(self.R_dtm, self.L, self.n_x)
        if self.n_y:
            self.y_values = np.linspace(0, self.L - self.R_dtm, self.n_y + 1)[1:]

        if self.n_alpha:
            self.alpha_values = (
                np.linspace(*self.alpha, self.n_alpha) if self.set_logscale == False
                else self.my_logspace(*self.alpha, self.n_alpha)
            )
            if self.n_y:
                self.n_x = self.n_alpha
                self.x_values = self.alpha_values
                self.w = self.params['rho']
            elif self.n_x:
                self.n_y = self.n_alpha
                self.y_values = self.alpha_values
                self.w = self.params['lambda']

    def _plot_annotations(self):
        self.xlabel = 'Inner Radius ($\\rho$)'
        self.ylabel = 'Thickness ($\\lambda$)'
        self.title = (
            'Diatom Flux under Bacterial Spherical Shells'
            + f'\n$\\tau_c/\\tau_d={self.T_ratio}$'
        )
        
        if self.n_alpha:
            abs_label = 'Timescale Ratios ($\\tau_c/\\tau_d$)'
            if self.iterate_radii:
                self.ylabel = abs_label
                self.title += f', $\\lambda={self.w}$'
            elif self.iterate_thick:
                self.xlabel = abs_label
                self.title += f',   $\\rho={self.w}$'