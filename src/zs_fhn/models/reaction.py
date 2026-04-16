import numpy as np


def reaction_terms(u, v, params):
    """
    Compute local reaction terms f(u,v), g(u,v) for FitzHugh–Nagumo model.

    This version implements the standard (non-conserved) FHN reaction terms.
    It is written to support both scalar inputs and NumPy arrays.

    Parameters
    ----------
    u : float or np.ndarray
        Activator variable
    v : float or np.ndarray
        Inhibitor variable
    params : dict
        Dictionary containing model parameters:
            - epsilon
            - a
            - b

    Returns
    -------
    f : same shape as u
        Time derivative contribution for u (reaction part)
    g : same shape as v
        Time derivative contribution for v (reaction part)
    """

    epsilon = params["epsilon"]
    a = params["a"]
    b = params["b"]

    # FHN reaction terms
    f = u - (u**3) / 3 - v
    g = epsilon * (u + a - b * v)

    return f, g