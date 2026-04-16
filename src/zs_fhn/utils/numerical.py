def laplacian(u, dx, dim=1, bc="neumann"):
    """
    Compute Laplacian of u.

    Parameters
    ----------
    u : np.ndarray
        1D or 2D array
    dx : float
        Grid spacing
    dim : int
        1 or 2
    bc : str
        "neumann" (default)
    """
    pass


def time_integrator(u, du_dt, dt):
    """
    Generic time integrator (e.g., Euler step).
    """
    pass