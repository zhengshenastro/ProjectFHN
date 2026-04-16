import numpy as np
from zs_fhn.models.reaction import reaction_terms

def u_nullcline(u, params):
    """
    Compute u-nullcline: du/dt = 0

    Returns:
        v as a function of u
    """
    return u - (u**3) / 3


def v_nullcline(u, params):
    """
    Compute v-nullcline: dv/dt = 0

    Returns:
        v as a function of u
    """
    a = params["a"]
    b = params["b"]

    return (u + a) / b


def vector_field(U, V, params):
    """
    Compute vector field (du/dt, dv/dt) on grid.

    Parameters
    ----------
    U, V : np.ndarray
        Meshgrid arrays

    Returns
    -------
    dU, dV : np.ndarray
    """

    dU, dV = reaction_terms(U, V, params)

    return dU, dV


def find_fixed_points(params):
    """
    Compute fixed points analytically via solving cubic equation.

    Returns:
        list of (u, v)
    """

    a = params["a"]
    b = params["b"]

    # coefficients of cubic: (b/3)u^3 + (1-b)u + a = 0
    coeffs = [
        b / 3,      # u^3
        0,          # u^2
        (1 - b),    # u
        a           # constant
    ]

    roots = np.roots(coeffs)

    fixed_points = []

    for r in roots:
        if np.isreal(r):
            u = np.real(r)
            v = (u + a) / b
            fixed_points.append((u, v))

    return fixed_points