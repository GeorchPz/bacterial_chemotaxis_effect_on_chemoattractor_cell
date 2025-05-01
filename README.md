# Diffusion-Consumption Simulation Framework (TFG)

This project contains the code developed for my Bachelor Thesis (Trabajo de Fin de Grado - TFG) focused on simulating and analysing diffusion-consumption problems (governed by the PDE bellow). The primary application involves modelling nutrient flux towards a central entity (e.g., a diatom) influenced by surrounding consuming entities (e.g., bacteria) under various spatial configurations.

$$
    \frac{\partial n (\vec{r}, t)}{\partial t} =
    \nabla^2 n (\vec{r}, t)
    - \alpha n(\vec{r}, t) c(\vec{r}, t)
$$

Where:
-   $n(\vec{r}, t)$ is the nutrient concentration at position $\vec{r}$ and time $t$.
-   $\alpha$ is the consumption rate.
-   $c(\vec{r}, t)$ is the concentration of the consuming entities (e.g., bacteria).


## Project Structure

### [`base/`](base)

This folder provides the foundational components and shared utilities for the simulation framework.

*   **[`base_plotter.py`](base\base_plotter.py):** Defines the [`BasePlotter`](base\base_plotter.py#L3) class. This class offers common plotting functionalities used throughout the project, such as setting up Matplotlib styles ([`configure_rc_params`](base\base_plotter.py#L7)), adding labels and titles ([`_set_plot_annotations`](base\base_plotter.py#L22)), adjusting figure layouts ([`_adjust_figure`](base\base_plotter.py#L28)), and saving plots ([`save`](base\base_plotter.py#L32)).
*   **[`base_fluxmap.py`](base\base_fluxmap.py):** Defines the [`BaseFluxMap`](base\base_fluxmap.py#L10) abstract base class. It establishes the interface for generating and visualizing "flux maps," which show how the nutrient flux onto the central entity varies across a parameter space (e.g., bacterial layer geometry, consumption rates). It leverages `joblib` for parallel computation ([`__parallel_solve`](base\base_fluxmap.py#L70)) to accelerate the process and includes methods for plotting the map ([`plot`](base\base_fluxmap.py#L168)) and finding flux minima ([`search_minimum`](base\base_fluxmap.py#L152)). Subclasses must implement specific details like parameter initialization ([`_init_values`](base\base_fluxmap.py#L20)) and plot annotations ([`_plot_annotations`](base\base_fluxmap.py#L33)).
*   **[`diffusion_plotter.py`](base\diffusion_plotter.py):** Defines the [`DiffusionPlotter`](base\diffusion_plotter.py#L7) class, inheriting from [`BasePlotter`](base\base_plotter.py#L3). This class is tailored for plotting specific outputs from individual simulations, such as nutrient concentration profiles ([`concentrations`](base\diffusion_plotter.py#L19)), the spatial distribution of nutrient flux ([`nutrient_flux`](base\diffusion_plotter.py#L41)), and the time evolution or steady-state value of the flux at the central entity's boundary ([`diatom_flux`](base\diffusion_plotter.py#L66)). It also includes helper methods for combined plots ([`double_plot`](base\diffusion_plotter.py#L93), [`triple_plot`](base\diffusion_plotter.py#L100)).
*   **[`__init__.py`](base\__init__.py):** Makes the key classes from this module easily importable and also imports `numpy` and `matplotlib.pyplot` for convenience.

### [`oneD/`](oneD)

This folder contains the implementations specific to one-dimensional diffusion-consumption problems.

*   **[`solver1D.py`](oneD\solver1D.py):** Defines the [`Solver1D`](oneD\solver1D.py#L5) class, which implements numerical methods (finite differences, using `scipy.linalg.solve_banded` for optimization ([`__solve_ode_opt`](oneD\solver1D.py#L101))) to solve the steady-state (ODE) diffusion-consumption equation in 1D ([`ODESolver`](oneD\solver1D.py#L51)). It also includes capabilities for solving the time-dependent (PDE) version using methods like `scipy.integrate.solve_ivp` ([`PDESolver`](oneD\solver1D.py#L113)). It calculates the resulting nutrient concentration profile (`n`) and flux (`flux`).
*   **[`solver1D_analyt.py`](oneD\solver1D_analyt.py):** Defines [`Solver1D_UniformBacterium`](oneD\solver1D_analyt.py#L5), a subclass of [`Solver1D`](oneD\solver1D.py#L5) that incorporates an analytical solution ([`AnalyticalSolver`](oneD\solver1D_analyt.py#L19)) for the specific case of a uniform bacterial distribution (`c_const`). It uses `sympy` for symbolic calculations.
*   **[`fluxmap1D.py`](oneD\fluxmap1D.py):** Defines the [`FluxMap1D`](oneD\fluxmap1D.py#L4) class, inheriting from [`BaseFluxMap`](base\base_fluxmap.py#L10). It provides the concrete implementation for generating flux maps in 1D, specifying [`Solver1D`](oneD\solver1D.py#L5) as the solver class and defining the relevant parameters (e.g., starting point `x0`, layer length `l`) and plot labels.
*   **[`__init__.py`](oneD\__init__.py):** Makes the solver and flux map classes from this module easily importable.

### [`threeD/`](threeD)

This folder contains the implementations specific to three-dimensional diffusion-consumption problems, assuming spherical symmetry.

*   **[`solver3D.py`](threeD\solver3D.py):** Defines the [`Solver3D`](threeD\solver3D.py#L5) class. It implements numerical methods (finite differences, optimized using `scipy.linalg.solve_banded` ([`__solve_ode_opt`](threeD\solver3D.py#L72))) to solve the steady-state (ODE) diffusion-consumption equation in spherical coordinates with spherical symmetry ([`SolveODE`](threeD\solver3D.py#L37)). It handles the spatial discretization ([`__discretise_system`](threeD\solver3D.py#L27)) and calculates the radial nutrient concentration profile (`n`) and flux (`flux`). It may also contain stubs or implementations for analytical solutions ([`SolveAnalytically`](threeD\solver3D.py#L108)).
*   **[`fluxmap3D.py`](threeD\fluxmap3D.py):** Defines the [`FluxMap3D`](threeD\fluxmap3D.py#L41) class, inheriting from [`BaseFluxMap`](base\base_fluxmap.py#L10). It provides the concrete implementation for generating flux maps in 3D (spherical), specifying [`Solver3D`](threeD\solver3D.py#L5) as the solver class. It defines parameters relevant to spherical shells (e.g., inner radius `r0`, thickness `lambda`, consumption time `Tc`) and sets appropriate plot annotations ([`_plot_annotations`](threeD\fluxmap3D.py#L43)).
*   **[`__init__.py`](threeD\__init__.py):** Makes the solver and flux map classes from this module easily importable.

## Jupyter Notebooks

The project includes several Jupyter notebooks used for experimentation, analysis, and visualization. They demonstrate the usage of the solvers and flux map classes.

Key analyses performed in the notebooks include:

*   Exploring the impact of relative diffusion and consumption rates (`Tc` vs. `Td`).
*   Testing various bacterial concentration profiles (constant, step, exponential, etc.) in 1D and 3D.
*   Generating and analysing flux maps based on bacterial layer geometry (e.g., shell thickness, inner radius) and consumption time.
*   Finding optimal parameters (e.g., geometry for maximum/minimum flux).
*   Applying the 3D model to real-world diatom and bacteria dimension data.

## Dependencies

*   NumPy
*   Matplotlib
*   SciPy
*   SymPy (for analytical solutions)
*   joblib (for parallel processing in flux maps)
*   tqdm (for progress bars)

Install dependencies using pip:
```bash
pip install numpy matplotlib scipy sympy joblib tqdm
```

## Usage

Refer to the `main1D.py`, `main3D.py` scripts and the Jupyter notebooks for examples on how to:

1.  Define system parameters (geometry, diffusion/consumption rates).
2.  Define bacterial concentration functions (`c_func`).
3.  Instantiate `Solver1D`/`Solver3D` or `FluxMap1D`/`FluxMap3D`.
4.  Run simulations using the `.solve()` methods.
5.  Visualize results using the `.plot` attribute (e.g., `S1D.plot.triple_plot()`, `FM3D.plot()`).