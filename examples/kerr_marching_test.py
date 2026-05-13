import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt

import simulator.config as config
config.jax_init()

import simulator.beam as beam_initialiser
import simulator.domain as d
import simulator.nonlinear_propagator as nlp


# -------------------------
# Basic domain
# -------------------------
n_cells = 128
Np = 20000

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

lengths = 2 * np.array([extent_x, extent_y, extent_z])
probing_extent = extent_z
probing_direction = "z"

zero_ne = np.zeros((n_cells, n_cells, n_cells))

domain = d.ScalarDomain(
    lengths,
    n_cells,
    ne=zero_ne,
    probing_direction=probing_direction,
)

# -------------------------
# Beam
# -------------------------
beam_size = 1.0e-3
divergence = 0.0
ne_extent = probing_extent

beam = beam_initialiser.Beam(
    Np,
    beam_size,
    divergence,
    ne_extent,
    probing_direction=probing_direction,
    beam_type="circular",
)

# -------------------------
# Kerr propagation power sweep
# -------------------------
powers = [0.0, 1e6, 1e7, 1e8, 1e9]

for power in powers:
    final_state, history = nlp.solve_kerr_marching(
        beam.s0,
        domain,
        probing_extent,
        power=power,
        n2=2.7e-20,
        n0=1.33,
        n_steps=256,
    )

    history = np.array(history)

    r_rms = np.sqrt(np.mean(history[:, 0, :]**2 + history[:, 1, :]**2, axis=1))
    z_axis = history[:, 2, 0]

    print(
        f"Power = {power:.1e} W | "
        f"Initial RMS = {r_rms[0]*1e3:.6f} mm | "
        f"Final RMS = {r_rms[-1]*1e3:.6f} mm | "
        f"Ratio = {r_rms[-1]/r_rms[0]:.6f}"
    )

    plt.figure()
    plt.plot(z_axis * 1e3, r_rms * 1e3)
    plt.xlabel("z / mm")
    plt.ylabel("RMS radius / mm")
    plt.title(f"Beam radius evolution, P = {power:.1e} W")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"kerr_rms_power_{power:.0e}.png", dpi=200)
    plt.close()