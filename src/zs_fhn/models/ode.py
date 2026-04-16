from .reaction import reaction_terms
import numpy as np
from numba import njit


def step_ode(u, v, params):
    """
    Compute time derivatives (du/dt, dv/dt) for ODE system.

    This is a thin wrapper around reaction_terms, keeping a unified interface
    for future extension (e.g., higher-order integrators).

    Parameters
    ----------
    u : float or np.ndarray
        Activator variable
    v : float or np.ndarray
        Inhibitor variable
    params : dict
        Model parameters

    Returns
    -------
    du_dt : same shape as u
    dv_dt : same shape as v
    """

    du_dt, dv_dt = reaction_terms(u, v, params)

    return du_dt, dv_dt

@njit
def _rk4_step(u, v, dt, epsilon, a, b):

    def f(u, v):
        return u - (u**3)/3 - v

    def g(u, v):
        return epsilon * (u + a - b*v)

    # k1
    k1_u = f(u, v)
    k1_v = g(u, v)

    # k2
    k2_u = f(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)
    k2_v = g(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)

    # k3
    k3_u = f(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)
    k3_v = g(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)

    # k4
    k4_u = f(u + dt*k3_u, v + dt*k3_v)
    k4_v = g(u + dt*k3_u, v + dt*k3_v)

    u_new = u + (dt/6.0)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
    v_new = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return u_new, v_new


@njit
def _simulate_ode_numba(u0, v0, dt, T, epsilon, a, b):
    n_steps = int(T / dt)

    u = np.zeros(n_steps)
    v = np.zeros(n_steps)

    u[0] = u0
    v[0] = v0

    for i in range(n_steps - 1):
        u[i+1], v[i+1] = _rk4_step(u[i], v[i], dt, epsilon, a, b)

    return u, v


def simulate_ode(u0, v0, params):
    """
    Numba-accelerated ODE solver.
    """

    dt = params["dt"]
    T = params["T"]

    epsilon = params["epsilon"]
    a = params["a"]
    b = params["b"]

    u, v = _simulate_ode_numba(u0, v0, dt, T, epsilon, a, b)

    t = np.linspace(0, T, len(u))

    return t, u, v




