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

    gx = grad_x[ix, iy]
    gy = grad_y[ix, iy]

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
):
    """
    March rays through a Kerr medium using self-consistent transverse
    intensity deposition.

    This is separate from propagator.solve so the normal synthPy
    static-domain solver is not affected.
    """

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
    )

    final_state, history = jax.lax.scan(
        lambda state, _: kerr_step(state, args),
        state0,
        xs=None,
        length=n_steps,
    )

    return final_state, history