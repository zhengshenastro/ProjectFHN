def default_params():
    """
    Return default parameter dictionary for FHN system.

    This includes both ODE and PDE parameters.
    """

    return {
        # ===== ODE parameters =====
        "epsilon": 0.05,
        "a": 0.7,
        "b": 0.8,

        # ===== time integration =====
        "dt": 0.01,
        "T": 150,   # 🔑 稍大，适合判断振荡

        # ===== PDE (spatial) =====
        "L": 100.0,   # domain size
        "Nx": 200,    # grid points

        # diffusion coefficients
        "Du": 0.1,
        "Dv": 0.05,
    }

def scan3d_default_grid():
    """
    Default parameter grid for 3D scan.

    Returns
    -------
    dict with arrays:
        a_vals, b_vals, eps_vals
    """

    import numpy as np

    return {
        "a_vals": np.linspace(0.4, 1.6, 8),
        "b_vals": np.linspace(0.5, 1.2, 6),
        "eps_vals": np.linspace(0.02, 0.2, 6),
    }

def update_params(params, updates):
    """
    Return a new params dict with updated values.
    """
    pass