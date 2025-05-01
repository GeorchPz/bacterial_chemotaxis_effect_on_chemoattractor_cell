from . import plt

class BasePlotter:
    def __init__(self):
        self.fig = None

    @staticmethod
    def configure_rc_params(titlesize=14, outersize=12, innersize=9):
        '''Configure the figure parameters applying a predefined style.'''
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': outersize,
            'axes.titlesize': titlesize,
            'axes.labelsize': outersize,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'legend.fontsize': innersize,
            'xtick.labelsize': innersize,
            'ytick.labelsize': innersize,
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5
        })

    @staticmethod
    def _set_plot_annotations(ax, xlabel, ylabel, title):
        '''Set the labels, title, legend, and grid for the given axis.'''
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(loc='best')
        ax.grid(True)

    @staticmethod
    def _adjust_figure():
        '''Final adjusts to make it look better'''
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35, left=0.05, right=0.95)

    def save(self, path, filename, dpi=300):
        '''Save the figure to a file with the given path, filename, and dpi.'''
        full_path = f"{path}/{filename}.png"
        self.fig.savefig(full_path, dpi=dpi, bbox_inches='tight')