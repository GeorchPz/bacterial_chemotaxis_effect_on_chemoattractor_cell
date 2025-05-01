from base import BaseFluxMap, np
from .solver3D import Solver3D

class FluxMap3D(BaseFluxMap):
    @property
    def _solver_class(self):
        return Solver3D

    def _init_values(self):
        # ODE parameters
        self.R_dtm = self.params['R_dtm']
        self.L     = self.params['L']
        self.Tc    = self.params['Tc']

        # FluxMap parameters
        self.n_x = self.params.get('n_r0')
        self.n_y = self.params.get('n_l')
        self.n_Tc = self.params.get('n_Tc')
        # params.get() returns None if the key is not found

        self.iterate_thick = 'n_l' in self.params
        self.iterate_radii = 'n_r0' in self.params

        if self.n_x:
            self.x_values = np.linspace(self.R_dtm, self.L, self.n_x)
        if self.n_y:
            self.y_values = np.linspace(0, self.L - self.R_dtm, self.n_y + 1)[1:]

        if self.n_Tc:

            self.Tc_values = (
                np.linspace(*self.Tc, self.n_Tc) if self.set_logscale == False
                else self.my_logspace(*self.Tc, self.n_Tc)
            )
            if self.n_y:
                self.n_x = self.n_Tc
                self.x_values = self.Tc_values
                self.w = self.params['r0']
            elif self.n_x:
                self.n_y = self.n_Tc
                self.y_values = self.Tc_values
                self.w = self.params['l']

    def _plot_annotations(self):
        self.xlabel = 'Inner Radius ($r_0$)'
        self.ylabel = 'Thickness ($\\lambda$)'
        self.title = (
            'Diatom Flux under Bacterial Spherical Shells'
            + f'\n$T_c={self.Tc}$'
        )
        
        if self.n_Tc:
            abs_label = 'Consumption Time ($T_c$)'
            if self.iterate_radii:
                self.ylabel = abs_label
                self.title += f', $\\lambda={self.w}$'
            elif self.iterate_thick:
                self.xlabel = abs_label
                self.title += f',   $r_0={self.w}$'