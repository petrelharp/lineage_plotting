#!/usr/bin/env python3
import os, sys
import tskit, pyslim
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import fe

usage = """
Plots the 1D density, along with a comparison to the fenics-computed numerical
solution.

Usage:
    plot_pme_density.py [name of treefile] [(start time) (end time) (number of steps)]
"""

if len(sys.argv) != 2 and len(sys.argv) != 5:
    raise ValueError(usage)

output_time = "pdf"

treefile = sys.argv[1]
assert treefile[-6:] == ".trees"
outbase = treefile[:-6]

input_times = (len(sys.argv) > 2)

if input_times:
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    num_steps = int(sys.argv[4])

ts = pyslim.load(treefile)
params = ts.metadata['SLiM']['user_metadata']
for n in params:
    if len(params[n]) == 1:
        params[n] = params[n][0]


dx = 0.8
T = params['RUNTIME']
dt = params["DT"]
max_gen = ts.metadata["SLiM"]["generation"] - 1
y_bins = [0, params['HEIGHT']]
x_bins = np.arange(0, params['WIDTH'] + dx, dx)
x_mids = x_bins[1:] - np.diff(x_bins)/2
t_bins = np.arange(0, max_gen + 1)

# Population sizes for each time step within each x bin
popsize = pyslim.population_size(ts, x_bins, y_bins, t_bins)

# "Observed" values of u
uhat = np.sum(popsize, axis=1) / (params["K"] * dx)
assert uhat.shape[0] == len(x_bins) - 1
assert uhat.shape[1] == len(t_bins) - 1

def plot_times(ax, times):
    cm = matplotlib.cm.cool
    steps = np.searchsorted(t_bins, max_gen - times / dt) - 1
    legend_steps = list(range(0, len(steps) + 1, int((len(steps) + 1) / 4)))
    for m, (j, t) in enumerate(zip(steps, times)):
        ax.plot(x_mids, uhat[:, j], c=cm(m / len(steps)),
                label=f"t={t:0.2f}" if m in legend_steps else None)
    ax.set_title(f"{min(times)} < t < {max(times)}")
    ax.set_xlabel("location")
    ax.set_ylabel("density")
    ax.legend()
    ax.axhline(0.0, ls=":")
    ax.axhline(1.0, ls=":")
    return

def add_theory(ax, times, theory):
    cm = matplotlib.cm.cool
    for m, t in enumerate(times):
        ax.plot(x_mids, theory[:, m], c=cm(m / len(times)), ls="--")
    return

if input_times:
    output_times = np.linspace(start_time, end_time, num_steps)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    plot_times(ax, output_times)
    theory = fe.pme(
            ts,
            output_times=output_times,
            x_bins=x_bins,
            sigma=params["DISPERSAL_SIGMA"],
            fenics_nx=101,
            fenics_ny=2,
            fenics_dt=0.05
    )
    add_theory(ax, output_times, theory)

    fig.savefig(f"{outbase}.steps.{output_type}")

else:

    #############

    plot_dt = 0.2
    plot_interval = 2
    t0 = [0, int(T/2) - plot_interval/2, T - plot_interval]
    fig, axes = plt.subplots(1, len(t0), figsize=(15,5))
    for t, ax in zip(t0, axes):
        tsteps = np.arange(t, t + plot_interval, plot_dt)
        plot_times(ax, tsteps)
        ax.legend()
        theory = fe.pme(
                ts,
                output_times=tsteps,
                x_bins=x_bins,
                sigma=params["DISPERSAL_SIGMA"],
                fenics_nx=101,
                fenics_ny=2,
                fenics_dt=0.05
        )
        add_theory(ax, tsteps, theory)


    fig.savefig(f"{outbase}.short_steps.{output_type}")


    #############

    output_times = np.arange(0, 6, 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    plot_times(ax, output_times)
    theory = fe.pme(
            ts,
            output_times=output_times,
            x_bins=x_bins,
            sigma=params["DISPERSAL_SIGMA"],
            fenics_nx=101,
            fenics_ny=2,
            fenics_dt=0.05
    )
    add_theory(ax, output_times, theory)

    fig.savefig(f"{outbase}.first_steps.{output_type}")

    #############

    output_times = np.arange(0, T, 10)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    plot_times(ax, output_times)
    theory = fe.pme(
            ts,
            output_times=output_times,
            x_bins=x_bins,
            sigma=params["DISPERSAL_SIGMA"],
            fenics_nx=101,
            fenics_ny=2,
            fenics_dt=0.05
    )
    add_theory(ax, output_times, theory)

    fig.savefig(f"{outbase}.long_steps.{output_type}")
