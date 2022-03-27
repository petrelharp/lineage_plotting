import fenics
import numpy as np

def basis_functions(V):
    u0 = fenics.Function(V)
    u0_vals = u0.vector()
    for j in range(V.dim()):
        a = np.zeros(V.dim())
        a[j] = 1
        u0_vals[:] = a
        yield u0


def project_locations(xy, V, K):
    """
    The basis functions are a partition of unity;
    this "spreads out 1/K units of mass" for each point
    across the basis functions.
    """
    one = fenics.Function(V)
    one.vector()[:] = 1.0
    init_vals = np.zeros(V.dim())
    for j, u0 in enumerate(basis_functions(V)):
        # the integral of u0:
        denom = fenics.assemble(one * u0 * fenics.dx)
        for p in xy:
            init_vals[j] += u0(*p) / K / denom
    u = fenics.Function(V)
    u.vector()[:] = init_vals
    return fenics.interpolate(u, V)


def mean_value(u, xmin, xmax, V):
    uu = fenics.Expression(
        f"((x[0] >= {xmin}) "
        f"&& (x[0] < {xmax}))"
        f"? 1 : 0;",
        element=V.ufl_element()
    )
    one = fenics.Function(V)
    one.vector()[:] = 1.0
    denom = fenics.assemble(uu * one * fenics.dx)
    if denom == 0:
        out = np.nan
    else:
        out = fenics.assemble(uu * u * fenics.dx) / denom
    return out


def pme(ts, output_times, x_bins, sigma, fenics_nx=101, fenics_ny=2, fenics_dt=0.05):
    '''
    Here "output_times" are in real, forwards-time units.
    '''
    width = ts.metadata['SLiM']['user_metadata']['WIDTH'][0]
    height = ts.metadata['SLiM']['user_metadata']['HEIGHT'][0]
    K = ts.metadata['SLiM']['user_metadata']['K'][0]
    theta = ts.metadata['SLiM']['user_metadata']['THETA'][0]
    slim_dt = ts.metadata['SLiM']['user_metadata']['DT'][0]

    step_ago = ts.metadata['SLiM']['generation'] - np.min(output_times) / slim_dt - 1
    init_xy = np.array([
        ts.individual(i).location[:2]
        for i in ts.individuals_alive_at(step_ago)
    ])

    # Create mesh and define function space
    mesh = fenics.RectangleMesh(
        p0=fenics.Point(0, 0),
        p1=fenics.Point(width, height),
        nx=fenics_nx,
        ny=fenics_ny,
    )
    V = fenics.FunctionSpace(mesh, 'P', 1)

    # Define initial value
    # Note that since out simulation is 1D, but the analytic solution is 2D,
    # to convert from the simulations' density-per-unit-x-area
    # to fenic's density-per-unit-xy-area we need to multiply the simulation
    # by 'height', which corresponds to placing height/K mass at each point.
    u_n = project_locations(init_xy, V, K/height)

    # Define variational problem
    u = fenics.Function(V)
    v = fenics.TestFunction(V)
    def get_F(u, v, dt):
        dx = fenics.dx
        F = (u * v * dx
             + dt * (sigma**2 / 2)
                * fenics.dot(fenics.grad(u**2), fenics.grad(v)) * dx
             - (1 / theta) * dt * u * (1 - u) * v * dx
             - u_n * v * dx
        )
        return F

    def observed(u, x_bins):
        """
        Note this is in units of (average per unit xy-area).
        """
        out = np.zeros(len(x_bins) - 1)
        for j in range(len(x_bins) - 1):
            out[j] = mean_value(u, x_bins[j], x_bins[j+1], V)
        return out

    # too much output!!!
    fenics.set_log_active(False)

    output = np.empty((len(x_bins) - 1, len(output_times)))
    # Time-stepping
    t = np.min(output_times)
    for j, next_t in enumerate(output_times):
        t_diff = next_t - t
        if t_diff > 0:
            num_dt = int(np.ceil(t_diff / fenics_dt))
            dt = t_diff / num_dt
            F = get_F(u, v, dt=dt)
            for _ in range(num_dt):
                # Compute solution
                fenics.solve(F == 0, u)
                # Update previous solution
                u_n.assign(u)
        output[:, j] = observed(u_n, x_bins)
        t = next_t

    return output
