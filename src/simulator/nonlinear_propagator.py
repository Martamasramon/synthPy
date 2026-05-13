import jax
import jax.numpy as jnp
from functools import partial


def deposit_intensity_xy(x, y, *, x_grid, y_grid, power_per_ray):
    """
    Deposit ray power onto a 2D transverse grid.

    Simple nearest-cell version. Good enough for first testing,
    but cloud-in-cell would be better later.
    """

    nx = x_grid.size
    ny = y_grid.size

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    ix = jnp.floor((x - x_grid[0]) / dx).astype(jnp.int32)
    iy = jnp.floor((y - y_grid[0]) / dy).astype(jnp.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    ix = jnp.clip(ix, 0, nx - 1)
    iy = jnp.clip(iy, 0, ny - 1)

    power_grid = jnp.zeros((nx, ny))
    power_grid = power_grid.at[ix, iy].add(power_per_ray * valid)

    intensity = power_grid / (dx * dy)

    return intensity

def kerr_step(state, args):
    """
    One z-marching Kerr update.

    state shape:
        (6, Np): x, y, z, vx, vy, vz

    For now this only updates positions and directions.
    """

    (
        x_grid,
        y_grid,
        dz,
        power_per_ray,
        n0,
        n2,
        domain,
        grad_x_static,
        grad_y_static,
        include_static,
    ) = args

    x = state[0, :]
    y = state[1, :]
    z = state[2, :]

    vx = state[3, :]
    vy = state[4, :]
    vz = state[5, :]

    theta_x = vx / vz
    theta_y = vy / vz

    intensity = deposit_intensity_xy(
        x,
        y,
        x_grid=x_grid,
        y_grid=y_grid,
        power_per_ray=power_per_ray,
    )

    dn_kerr = n2 * intensity

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    grad_x, grad_y = jnp.gradient(dn_kerr, dx, dy)

    # Interpolate gradient back to ray positions.
    # For first version, use nearest cell.
    ix = jnp.floor((x - x_grid[0]) / dx).astype(jnp.int32)
    iy = jnp.floor((y - y_grid[0]) / dy).astype(jnp.int32)

    ix = jnp.clip(ix, 0, x_grid.size - 1)
    iy = jnp.clip(iy, 0, y_grid.size - 1)

    gx_kerr = grad_x[ix, iy]
    gy_kerr = grad_y[ix, iy]
    
    if include_static:
        gx_static, gy_static = sample_static_gradient_nearest(
            x, y, z, domain, grad_x_static, grad_y_static
        )
        gx = gx_static + gx_kerr
        gy = gy_static + gy_kerr
    else:
        gx = gx_kerr
        gy = gy_kerr

    theta_x = theta_x + dz * gx / n0
    theta_y = theta_y + dz * gy / n0

    x = x + dz * theta_x
    y = y + dz * theta_y
    z = z + dz

    # Convert slopes back into velocity-like components.
    # Keep speed approximately fixed.
    vz_new = vz
    vx_new = theta_x * vz_new
    vy_new = theta_y * vz_new

    new_state = state.at[0, :].set(x)
    new_state = new_state.at[1, :].set(y)
    new_state = new_state.at[2, :].set(z)
    new_state = new_state.at[3, :].set(vx_new)
    new_state = new_state.at[4, :].set(vy_new)
    new_state = new_state.at[5, :].set(vz_new)

    return new_state, new_state

def solve_kerr_marching(
    s0,
    domain,
    probing_depth,
    *,
    power,
    n2=2.7e-20,
    n0=1.33,
    n_steps=256,
    include_static=True,
    lwl=1064e-9,
):
    """
    March rays through a Kerr medium using self-consistent transverse
    intensity deposition.

    This is separate from propagator.solve so the normal synthPy
    static-domain solver is not affected.
    """
    
    if include_static:
        grad_x_static, grad_y_static = precompute_static_transverse_gradients(
        domain,
        lwl=lwl,
        )
    else:
        grad_x_static = None
        grad_y_static = None

    dz = probing_depth / n_steps
    power_per_ray = power / s0.shape[1]

    x_grid = domain.x
    y_grid = domain.y

    state0 = s0[:6, :]

    args = (
        x_grid,
        y_grid,
        dz,
        power_per_ray,
        n0,
        n2,
        domain,
        grad_x_static,
        grad_y_static,
        include_static,
    )

    final_state, history = jax.lax.scan(
        lambda state, _: kerr_step(state, args),
        state0,
        xs=None,
        length=n_steps,
    )

    return final_state, history

def precompute_static_transverse_gradients(domain, *, lwl=1064e-9):
    """
    Convert the existing synthPy ScalarDomain into static transverse
    refractive-index gradients.

    For z-propagation, we need ∂n/∂x and ∂n/∂y as functions of x,y,z.
    """

    import jax.numpy as jnp
    from scipy.constants import c

    dx = domain.x[1] - domain.x[0]
    dy = domain.y[1] - domain.y[0]

    if domain.edensity:
        omega = 2.0 * jnp.pi * c / lwl

        # Same critical-density scaling used in synthPy propagator.
        nc = 3.14207787e-4 * omega**2

        # Plasma refractive index approximation.
        # For weak plasma: n ≈ 1 - 0.5 ne/nc
        n_static = 1.0 - 0.5 * domain.ne / nc
    else:
        n_static = domain.refrac_field

    grad_x = jnp.gradient(n_static, dx, axis=0)
    grad_y = jnp.gradient(n_static, dy, axis=1)

    return grad_x, grad_y

def sample_static_gradient_nearest(x, y, z, domain, grad_x_static, grad_y_static):
    """
    Sample precomputed static ∂n/∂x and ∂n/∂y at ray positions.

    First version uses nearest-cell interpolation. Later this should become
    trilinear interpolation.
    """

    import jax.numpy as jnp

    dx = domain.x[1] - domain.x[0]
    dy = domain.y[1] - domain.y[0]
    dz = domain.z[1] - domain.z[0]

    ix = jnp.floor((x - domain.x[0]) / dx).astype(jnp.int32)
    iy = jnp.floor((y - domain.y[0]) / dy).astype(jnp.int32)
    iz = jnp.floor((z - domain.z[0]) / dz).astype(jnp.int32)

    ix = jnp.clip(ix, 0, domain.x.size - 1)
    iy = jnp.clip(iy, 0, domain.y.size - 1)
    iz = jnp.clip(iz, 0, domain.z.size - 1)

    gx_static = grad_x_static[ix, iy, iz]
    gy_static = grad_y_static[ix, iy, iz]

    return gx_static, gy_static

def solve(
    beam,
    domain,
    probing_depth,
    *,
    power,
    n2=2.7e-20,
    n0=1.33,
    n_steps=256,
    include_static=True,
    return_raw_results=False,
    return_E=False,
    lwl=1064e-9,
):
    """
    synthPy-style wrapper for the nonlinear Kerr marching solver.

    This lets simulator.propagator.solve(..., nonlinear=True) dispatch here.
    """

    import time
    import jax.numpy as jnp

    start = time.time()

    assert beam.shape[0] == 9, "Expected Beam.s0 with shape (9, Np)."

    final_state_6, history = solve_kerr_marching(
        beam,
        domain,
        probing_depth,
        power=power,
        n2=n2,
        n0=n0,
        n_steps=n_steps,
        include_static=include_static,
        lwl=lwl,
    )

    duration = time.time() - start

    # Rebuild a 9 x Np ray state so downstream synthPy-style code has
    # positions, velocities, amplitude, phase and polarisation slots.
    final_state_9 = jnp.array(beam)
    final_state_9 = final_state_9.at[:6, :].set(final_state_6)

    if return_raw_results:
        return final_state_9, history, duration

    from shared.propagation import ray_to_Jonesvector

    rf, Jf = ray_to_Jonesvector(
        final_state_9,
        probing_depth,
        probing_direction=domain.probing_direction,
        return_E=return_E,
    )

    return rf, Jf, duration