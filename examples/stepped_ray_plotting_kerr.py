import numpy as np
import matplotlib.pyplot as plt

import sys

#add path
sys.path.insert(0, '../')     # import path/to/synthpy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dims", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-s", "--samples", type = int)
parser.add_argument("-n", "--stepResolution", type = int)
args = parser.parse_args()

n_cells = 128
if args.dims is not None:
    n_cells = args.dims

Np = 100000
if args.rays is not None:
    Np = args.rays

samples = 64
if args.samples is not None:
    samples = args.samples

stepResolution = 128
if args.stepResolution is not None:
    stepResolution = args.stepResolution

import src.simulator.config as config
config.jax_init()

import simulator.beam as beam_initialiser
import simulator.domain as d
import simulator.propagator as p
import processing.diagnostics as diag

import importlib
importlib.reload(beam_initialiser)
importlib.reload(d)
importlib.reload(p)
importlib.reload(diag)

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

#x = np.linspace(-extent_x, extent_x, n_cells)
#y = np.linspace(-extent_y, extent_y, n_cells)
#z = np.linspace(-extent_z, extent_z, n_cells)

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

#domain = d.ScalarDomain(x = x, y = y, z = z, extent = probing_extent, probing_direction = probing_direction)     # create domain
# Much simpler domain function, no longer needlessly takes in beam values, they are fully seperated
domain = d.ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos", probing_direction = probing_direction) # B_on = False by default

lwl = 1064e-9 #define laser wavelength

# initialise beam
divergence = 5e-5   # realistic divergence value
beam_size = extent_x    # beam radius
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
beam_type = 'circular'

beam_definition = beam_initialiser.Beam(
    Np, beam_size, divergence, ne_extent,
    probing_direction = probing_direction,
    beam_type = "circular"
)

rf, history, duration = p.solve(
    beam_definition.s0,
    domain,
    probing_extent,
    nonlinear=True,
    kerr_power=1e9,
    kerr_n2=2.7e-20,
    kerr_n0=1.33,
    kerr_steps=stepResolution,
    return_raw_results=True,
)
from processing.plotting import stepped_ray_plot
import numpy as np
import matplotlib.pyplot as plt

history_np = np.array(history)

sample_indices = np.random.choice(
    history_np.shape[2],
    size=min(samples, history_np.shape[2]),
    replace=False,
)

z = history_np[:, 2, sample_indices] * 1e3
x = history_np[:, 0, sample_indices] * 1e3
y = history_np[:, 1, sample_indices] * 1e3

plt.figure()
plt.plot(z, x, alpha=0.3)
plt.xlabel("z / mm")
plt.ylabel("x / mm")
plt.title("Kerr + static domain: x-z ray paths")
plt.grid(True)
plt.tight_layout()
plt.savefig("stepped_ray_plotting_kerr_xz.png", dpi=200)
plt.close()

plt.figure()
plt.plot(z, y, alpha=0.3)
plt.xlabel("z / mm")
plt.ylabel("y / mm")
plt.title("Kerr + static domain: y-z ray paths")
plt.grid(True)
plt.tight_layout()
plt.savefig("stepped_ray_plotting_kerr_yz.png", dpi=200)
plt.close()

r_rms = np.sqrt(np.mean(history_np[:, 0, :]**2 + history_np[:, 1, :]**2, axis=1))
z_axis = history_np[:, 2, 0]

plt.figure()
plt.plot(z_axis * 1e3, r_rms * 1e3)
plt.xlabel("z / mm")
plt.ylabel("RMS beam radius / mm")
plt.title("Kerr + static domain: RMS radius")
plt.grid(True)
plt.tight_layout()
plt.savefig("stepped_ray_plotting_kerr_rms.png", dpi=200)
plt.close()

print("Initial RMS radius / mm:", r_rms[0] * 1e3)
print("Final RMS radius / mm:", r_rms[-1] * 1e3)
print("Final / initial:", r_rms[-1] / r_rms[0])
print("Saved Kerr plots.")