import numpy as np


# =========================
# Laplacian (1D, Neumann)
# =========================
def laplacian_1d(u, dx):
    """
    1D Laplacian with Neumann boundary condition.
    """

    u_ext = np.zeros(len(u) + 2)

    u_ext[1:-1] = u

    # Neumann BC: du/dx = 0
    u_ext[0] = u[0]
    u_ext[-1] = u[-1]

    return (u_ext[2:] - 2*u_ext[1:-1] + u_ext[:-2]) / dx**2


# =========================
# RHS (standard FHN PDE)
# =========================
def fhn_pde_rhs_standard(u, v, params, dx):
    """
    RHS for standard FHN PDE.
    """

    epsilon = params["epsilon"]
    a = params["a"]
    b = params["b"]

    Du = params.get("Du", 1.0)
    Dv = params.get("Dv", 0.0)

    # diffusion
    lap_u = laplacian_1d(u, dx)
    lap_v = laplacian_1d(v, dx)

    # reaction
    f = u - (u**3)/3 - v
    g = epsilon * (u + a - b*v)

    du_dt = Du * lap_u + f
    dv_dt = Dv * lap_v + g

    return du_dt, dv_dt


# =========================
# One step (standard)
# =========================
def step_standard(u, v, params):
    """
    One RK4 step for standard FHN PDE.
    """

    dt = params["dt"]
    dx = params["dx"]

    # k1
    du1, dv1 = fhn_pde_rhs_standard(u, v, params, dx)

    # k2
    du2, dv2 = fhn_pde_rhs_standard(
        u + 0.5 * dt * du1,
        v + 0.5 * dt * dv1,
        params, dx
    )

    # k3
    du3, dv3 = fhn_pde_rhs_standard(
        u + 0.5 * dt * du2,
        v + 0.5 * dt * dv2,
        params, dx
    )

    # k4
    du4, dv4 = fhn_pde_rhs_standard(
        u + dt * du3,
        v + dt * dv3,
        params, dx
    )

    u_next = u + dt * (du1 + 2*du2 + 2*du3 + du4) / 6
    v_next = v + dt * (dv1 + 2*dv2 + 2*dv3 + dv4) / 6

    return u_next, v_next


# =========================
# One step (conserved)
# =========================
def step_conserved(u, v, params):
    """
    Placeholder for mass-conserved FHN PDE.
    """

    raise NotImplementedError(
        "Mass-conserved FHN PDE is not implemented yet."
    )


# =========================
# Simulation (dispatcher)
# =========================
def simulate_pde(u0, v0, params, model="standard"):
    """
    Simulate PDE system.

    Args:
        model: "standard" or "conserved"

    Returns:
        u_hist, v_hist
        shape = (time, space)
    """

    T = params["T"]
    dt = params["dt"]

    steps = int(T / dt)

    u = u0.copy()
    v = v0.copy()

    u_hist = []
    v_hist = []

    for _ in range(steps):

        if model == "standard":
            u, v = step_standard(u, v, params)

        elif model == "conserved":
            u, v = step_conserved(u, v, params)

        else:
            raise ValueError("Unknown model type")

        u_hist.append(u.copy())
        v_hist.append(v.copy())

    return np.array(u_hist), np.array(v_hist)



