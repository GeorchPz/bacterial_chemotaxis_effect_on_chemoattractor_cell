from base import BaseFluxMap, np
from .solver3D import Solver3D

def c_none(x):
    'Dummy function for no bacterial step distribution'
    return None

class FluxMap3D(BaseFluxMap):
    @property
    def _solver_class(self):
        return Solver3D

    def _init_values(self, set_logscale=False):

        self.set_logscale = set_logscale # Logscale for T_ratio values
        # ODE parameters
        self.R_dtm = self.params['R_dtm']
        self.L     = self.params['L']
        if 'alpha' in self.params:
            self.alpha = self.params['alpha']
            self.n_abs = self.params.get('n_alpha')
            if self.n_abs is None:
                self.T_ratio = 8*np.pi*self.L / self.alpha
            else:
                self.T_ratio = tuple(8*np.pi*self.L / np.array(self.alpha))
        elif 'T_ratio' in self.params:
            self.T_ratio = self.params['T_ratio']
            self.n_abs = self.params.get('n_T_ratio')
            if self.n_abs is None:
                self.alpha = 8*np.pi*self.L / self.T_ratio
            else:
                self.alpha = tuple(8*np.pi*self.L / np.array(self.T_ratio))

        # FluxMap parameters
        self.n_x = self.params.get('n_rho')
        self.n_y = self.params.get('n_lambda')
        # and self.n_abs
    
        # Check if we are iterating over thickness or radii
        self.iterate_thick = 'n_lambda' in self.params
        self.iterate_radii = 'n_rho' in self.params

        if self.n_x:
            # rho values
            self.x_values = np.linspace(self.R_dtm, self.L, self.n_x)
        if self.n_y:
            # lambda values
            self.y_values = np.linspace(0, self.L - self.R_dtm, self.n_y + 1)[1:]

        if self.n_abs:
            self.T_ratio_values = (
                np.linspace(*self.T_ratio, self.n_abs) if self.set_logscale == False
                else self.my_logspace(*self.T_ratio, self.n_abs)
            )
            if self.n_y:
                self.n_x = self.n_abs
                self.x_values = self.T_ratio_values
                self.w = self.params['rho']
            elif self.n_x:
                self.n_y = self.n_abs
                self.y_values = self.T_ratio_values
                self.w = self.params['lambda']

    def _plot_annotations(self):
        self.xlabel = 'Inner Radius ($\\rho$)'
        self.ylabel = 'Thickness ($\\lambda$)'
        self.cbar_label = '$\\Phi_D(\\rho, \\lambda)$'
        self.title = (
            'Diatom Flux under Bacterial Spherical Shells'
            + f'\n$\\tau_c/\\tau_d={self.T_ratio}$'
        )
        
        if self.n_abs:
            abs_label = 'Timescale Ratios ($\\tau_c/\\tau_d$)'
            if self.iterate_radii:
                self.ylabel = abs_label
                self.title += f', $\\lambda={self.w}$'
            elif self.iterate_thick:
                self.xlabel = abs_label
                self.title += f',   $\\rho={self.w}$'