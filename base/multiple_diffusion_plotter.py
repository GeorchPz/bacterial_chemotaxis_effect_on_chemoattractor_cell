from matplotlib.cm import GnBu, RdPu

from . import plt
from . import np
from .base_plotter import BasePlotter


class MultiDiffusionPlotter(BasePlotter):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        self.r_str = 'r'

        self.configure_rc_params()

    def concentrations(self, n_list, labels, ax=None):
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(5, 5))
        
        colours = GnBu(np.linspace(0.3, 0.7, len(n_list)))

        for i, n in enumerate(n_list):
            ax.plot(
                solver.r, n, color=colours[i],
                label=labels[i]
                )

        ax.set_xlabel('r')
        ax.set_ylabel('n(r)')

        ax_b = ax.twinx()
        ax_b.plot(solver.r, solver.c, 'r--', label='$c_{'+solver.c_tag+'}(r)$')
        ax_b.set_ylabel(f'c(r)', color='red')
        ax_b.tick_params(axis='y', labelcolor='red')

        # title = 'Nutrient Diffusion Given a Bacterial Distribution'
        self._set_plot_annotations(
            ax, self.r_str, f'n({self.r_str},t)', None)
        
        # Add the legend for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_b.get_legend_handles_labels()
        ax.legend(lines1 + [lines2[0]], labels1 + [labels2[0]], loc='best')


    def nutrient_flux(self, f_list, labels, ax=None):
        solver = self.parent # Parent's alias

        if ax is None:
            self.fig, ax = plt.subplots(figsize=(5, 5))

        colours = GnBu(np.linspace(0.3, 0.7, len(f_list)))
        colours_point = RdPu(np.linspace(0.3, 0.7, len(f_list)))

        for i, f in enumerate(f_list):
            ax.plot(
                solver.r, f, color=colours[i],
                label=labels[i]
                )

            # Last value plot
            diatom_flux_str = '$|\\Phi_{'+labels[i]+'}(r=R_D)|$' + f' = {f[0]:.6f}'
            ax.plot(
                solver.r[0], f[0], marker='o', linestyle='None', 
                color=colours_point[i],
                label= diatom_flux_str
                )
            
            # title = 'Absolute Nutrient Flux'
            self._set_plot_annotations(ax, self.r_str, '$|\\Phi'+f'({self.r_str})|$', None)

    def double_plot(
            self, n_list, f_list, labels,
            xlim=None
            ):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(2*5 + 1, 5))
        self._adjust_figure()

        self.concentrations(n_list, labels, self.axes[0])
        self.nutrient_flux(f_list, labels, self.axes[1])

        if xlim is not None:
            if len(xlim) == 2:
                for ax in self.axes:
                    ax.set_xlim(*xlim)
            else: 
                for ax in self.axes:
                    ax.set_xlim(0, xlim)