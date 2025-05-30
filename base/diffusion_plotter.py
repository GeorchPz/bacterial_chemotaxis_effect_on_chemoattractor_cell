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
            self.colour1 = 'green'
            self.colour2 = 'magenta'
        
        else: # 1D
            self.colour1 = 'yellow'
            self.colour2 = 'magenta'

        self.configure_rc_params()

    def concentrations(self, ax=None, legend_loc='best'):
        '''Plot the nutrient concentration n(x, t) and the bacterial concentration c(x).'''
        solver = self.parent # Parent's alias
        
        if ax is None:
            self.fig, ax = plt.subplots(figsize=(5, 5))
        
        if hasattr(solver.ode, 'n'):
            ax.plot(
                solver.x, solver.ode.n, color=self.colour1, linewidth=3,
                label='Steady'
                )
        
        if hasattr(solver.ode, 'analyt'):
            if hasattr(solver.ode.analyt, 'n'):
                ax.plot(
                    solver.x, solver.ode.analyt.n, color=self.colour2, linestyle='--',
                    label='Analyt.'
                )
        
        if hasattr(solver.pde, 'n'):
            # Always use exactly 10 time points
            time_indices = np.linspace(0, solver.nt-1, 10, dtype=int)
            # Use a wider color range or a different colormap for better distinction
            colours = viridis(np.linspace(0.2, 0.8, 10))
            for idx, i in enumerate(time_indices):
                ax.plot(
                solver.x, solver.pde.n[i], color=colours[idx], linestyle=':',
                label=f't = {solver.t[i]:.2f}'
                )
        
        # title = 'Nutrient Diffusion Given a Bacterial Distribution'
        self._set_plot_annotations(ax, self.x_str, f'n({self.x_str},t)', None
            )
        
        # Define the secondary y-axis for the bacteria concentration
        ax_b = ax.twinx()
        ax_b.plot(solver.x, solver.c, 'r', label='$c_{'+solver.c_tag+'}$'+f'({self.x_str})')
        ax_b.set_ylabel(f'c({self.x_str})', color='red')
        ax_b.tick_params(axis='y', labelcolor='red')

        # Add the legend for both axes
        if legend_loc == 'outside':
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_b.get_legend_handles_labels()
            ax.legend(
                lines + [lines2[0]],
                labels + [labels2[0]],
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                frameon=True
            )
        else:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_b.get_legend_handles_labels()
            ax.legend(lines + [lines2[0]], labels + [labels2[0]], loc=legend_loc)

    def nutrient_flux(self, ax=None, legend_loc='best'):
        '''Plot the absolute nutrient flux |Φ(x, t)| for dynamic-state and steady-state.'''
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(5, 5))
        
        if hasattr(solver.ode, 'abs_flux'):
            ax.plot(
                solver.x, solver.ode.abs_flux, color=self.colour1, linewidth=3,
                label='Steady'
                )
            # Last value plot
            diatom_flux_str = '$|\\Phi'+f'({self.x_str}=0)|$\n= {solver.ode.abs_flux[0]:.4f}'
            ax.plot(solver.x[0], solver.ode.abs_flux[0], 'ro', label= diatom_flux_str)
        
        if hasattr(solver.ode, 'analyt'):
            if hasattr(solver.ode.analyt, 'abs_flux'):
                ax.plot(
                    solver.x, solver.ode.analyt.abs_flux, color=self.colour2, linestyle='--',
                    label='Steady Analyt.'
                    )
        
        if hasattr(solver.pde, 'abs_flux'):
            colours = viridis(np.linspace(0.2, 0.8, solver.nt // 10))
            for idx, i in enumerate(range(0, solver.nt, solver.nt // 10 + 1)):
                ax.plot(
                    solver.x, solver.pde.abs_flux[i], color=colours[idx], linestyle=':',
                    label=f't = {solver.t[i]:.2f}'
                )
                
        # title = 'Absolute Nutrient Flux'
        self._set_plot_annotations(
            ax, self.x_str, '$|\\Phi'+f'({self.x_str},t)|$', None
            )
        
        # Add the legend for both axes
        if legend_loc == 'outside':
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(
                lines, labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                frameon=True
            )

    def diatom_flux(self, ax=None, legend_loc='best'):
        '''Plot the absolute flux received by the diatom at x=0 |Φ(x=0, t)|.'''
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(5, 5))

        ax.plot(solver.t, solver.pde.abs_flux_at_x0, 'b', label=f'$|\\Phi({self.x_str}=0, t)|$')
        
        diatom_flux_str = f'$|\\Phi({self.x_str}=0, t={solver.t[-1]:.2f})|$\n= {solver.pde.abs_flux[-1][0]:.6f}'
        ax.plot(solver.t[-1], solver.pde.abs_flux_at_x0[-1], 'ro',label=diatom_flux_str)
        
        if hasattr(solver.ode, 'abs_flux'):
            diatom_flux_steady = '$|\\Phi_{Steady}' + f'({self.x_str}=0)|$\n= {solver.ode.abs_flux[0]:.6f}'
            ax.axhline(solver.ode.abs_flux[0], color=self.colour1, label=diatom_flux_steady)
        
        # title = 'Diatom Flux'
        self._set_plot_annotations(ax, 't', f'$\\Phi_D(t) \\equiv |\\Phi({self.x_str}=0, t)|$', None)

        # Add the legend for both axes
        if legend_loc == 'outside':
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(
                lines, labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                frameon=True
            )

    def double_plot(self, xlim=None, legend_loc='best'):
        'Plot the nutrient concentration, and the nutrient flux'
        self.fig, self.axes = plt.subplots(1, 2, figsize=(2*5+1, 5))
        self._adjust_figure()
        
        self.concentrations(self.axes[0], legend_loc=legend_loc)
        self.nutrient_flux(self.axes[1], legend_loc=legend_loc)
        
        if xlim is not None:
            if len(xlim) == 2:
                for ax in self.axes:
                    ax.set_xlim(*xlim)
            else: 
                for ax in self.axes:
                    ax.set_xlim(0, xlim)


    def triple_plot(self, xlim=None, legend_loc='best'):
        'Plot the nutrient concentration, the nutrient flux, and the flux at x=0'
        self.fig, self.axes = plt.subplots(1, 3, figsize=(3*5+2, 5))
        self._adjust_figure()

        self.concentrations(self.axes[0], legend_loc)
        self.nutrient_flux(self.axes[1], legend_loc)
        self.diatom_flux(self.axes[2], legend_loc)
        
        if xlim is not None:
            if len(xlim) == 2:
                for ax in self.axes:
                    ax.set_xlim(*xlim)
            else: 
                for ax in self.axes:
                    ax.set_xlim(0, xlim)                    