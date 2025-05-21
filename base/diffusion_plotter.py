from matplotlib.cm import viridis

from . import plt
from . import np
from .base_plotter import BasePlotter


class DiffusionPlotter(BasePlotter):
    def __init__(self, parent, x_str='x'):
        super().__init__()
        self.parent = parent
        
        self.x_str = x_str
        if x_str == 'r': # 3D
            # Use of the radial coordinate r
            self.parent.x = self.parent.r
            # Analytical independent variable
            R_ext = self.parent.R_dtm/100
            nr_analyt = self.parent.nr//100
            self.parent.x_analyt = np.linspace(
                self.parent.r[0], R_ext, nr_analyt
            )
        else: # 1D
            # Analytical independent variable
            self.parent.x_analyt = np.linspace(
                self.parent.x[0], self.parent.x[-1], self.parent.nr/10
            )

        self.configure_rc_params()

    def concentrations(self, ax=None):
        '''Plot the nutrient concentration n(x, t) and the bacterial concentration c(x).'''
        solver = self.parent # Parent's alias
        
        if ax is None:
            self.fig, ax = plt.subplots(figsize=(6, 6))
        
        if hasattr(solver.pde, 'n'):
            # Always use exactly 10 time points
            time_indices = np.linspace(0, solver.nt-1, 10, dtype=int)
            # Use a wider color range or a different colormap for better distinction
            colours = viridis(np.linspace(0, 1, 10))
            for idx, i in enumerate(time_indices):
                ax.plot(
                solver.x_analyt, solver.pde.n[i],
                label=f'n({self.x_str}, t={solver.t[i]:.2f})', color=colours[idx]
                )
        
        if hasattr(solver.ode, 'n'):
            ax.plot(solver.x, solver.ode.n, 'g--', label='$n_{Steady}'+f'({self.x_str})$')
        
        if hasattr(solver.ode, 'analyt'):
            if hasattr(solver.ode.analyt, 'n'):
                ax.plot(solver.x, solver.ode.analyt.n, 'm:', label='$n_{Analytical}'+f'({self.x_str})$')
        
        self._set_plot_annotations(
            ax, self.x_str, f'Nutrient Concentration n({self.x_str},t)',
            'Nutrient Diffusion Given a Bacterial Distribution')
        
        # Define the secondary y-axis for the bacteria concentration
        ax_b = ax.twinx()
        ax_b.plot(solver.x, solver.c, 'r--', label='$c_{'+solver.c_tag+'}$'+f'({self.x_str})')
        ax_b.set_ylabel(f'Bacteria Concentration c({self.x_str})', color='red')
        ax_b.tick_params(axis='y', labelcolor='red')

        # Add the legend for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_b.get_legend_handles_labels()
        ax.legend(lines + [lines2[0]], labels + [labels2[0]], loc='best')

    def nutrient_flux(self, ax=None):
        '''Plot the absolute nutrient flux |Φ(x, t)| for dynamic-state and steady-state.'''
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(6, 6))
        
        if hasattr(solver.pde, 'abs_flux'):
            colours = viridis(np.linspace(0, 1, solver.nt // 10))
            for idx, i in enumerate(range(0, solver.nt, solver.nt // 10 + 1)):
                ax.plot(
                    solver.x_analyt, solver.pde.abs_flux[i],
                    label=f'$|\\Phi({self.x_str}, t={solver.t[i]:.2f})|$', color=colours[idx]
                )
        
        if hasattr(solver.ode, 'abs_flux'):
            ax.plot(solver.x, solver.ode.abs_flux, 'g--', label='$|\\Phi_{Steady}'+f'({self.x_str})|$')
            
            # Last value plot
            diatom_flux_str = '$|\\Phi_{Steady}'+f'({self.x_str}=0)|$ ={solver.ode.abs_flux[0]:.6f}'
            ax.plot(solver.x[0], solver.ode.abs_flux[0], 'ro', label= diatom_flux_str)
        
        if hasattr(solver.ode, 'analyt'):
            if hasattr(solver.ode.analyt, 'abs_flux'):
                ax.plot(solver.x, solver.ode.analyt.abs_flux, 'm:', label='$|\\Phi_{Analytical}'+f'({self.x_str})|$')
                
        self._set_plot_annotations(ax, self.x_str, 'Absolute Flux $|\\Phi'+f'({self.x_str})|$', 'Absolute Nutrient Flux')

    def diatom_flux(self, ax=None):
        '''Plot the absolute flux received by the diatom at x=0 |Φ(x=0, t)|.'''
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(solver.t, solver.pde.abs_flux_at_x0, 'b-', label=f'$|\\Phi({self.x_str}=0, t)|$')
        
        diatom_flux_str = f'$|\\Phi({self.x_str}=0, t={solver.t[-1]:.2f})|$ = {solver.pde.abs_flux[-1][0]:.6f}'
        ax.plot(solver.t[-1], solver.pde.abs_flux_at_x0[-1], 'ro',label=diatom_flux_str)
        
        if hasattr(solver.ode, 'abs_flux'):
            diatom_flux_steady = '$|\\Phi_{Steady}' + f'({self.x_str}=0)|$ ={solver.ode.abs_flux[0]:.6f}'
            ax.axhline(solver.ode.abs_flux[0], color='g', linestyle='--', label=diatom_flux_steady)
        
        self._set_plot_annotations(ax, 't', f'Absolute Flux $|\\Phi({self.x_str}=0, t)|$', 'Diatom Flux')

    def double_plot(self, xlim=None):
        'Plot the nutrient concentration, and the nutrient flux'
        self.fig, self.axes = plt.subplots(1, 2, figsize=(2*6+1, 6))
        self._adjust_figure()
        
        self.concentrations(self.axes[0])
        self.nutrient_flux(self.axes[1])
        
        if xlim is not None:
            if len(xlim) == 2:
                for ax in self.axes:
                    ax.set_xlim(*xlim)
            else: 
                for ax in self.axes:
                    ax.set_xlim(0, xlim)


    def triple_plot(self, xlim=None):
        'Plot the nutrient concentration, the nutrient flux, and the flux at x=0'
        self.fig, self.axes = plt.subplots(1, 3, figsize=(3*6+2, 6))
        self._adjust_figure()

        self.concentrations(self.axes[0])
        self.nutrient_flux(self.axes[1])
        self.diatom_flux(self.axes[2])
        
        if xlim is not None:
            if len(xlim) == 2:
                for ax in self.axes:
                    ax.set_xlim(*xlim)
            else: 
                for ax in self.axes:
                    ax.set_xlim(0, xlim)