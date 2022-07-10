#!/usr/bin/env python3

import sys
import tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

usage = """
Makes a plot of the location of individuals present in the tree sequence
at the provided time ago (defaults to zero).

Usage:
    {} (treefile) [time ago=0] [format=png] [max dimension=8]
""".format(sys.argv[0])

if len(sys.argv) < 2 or len(sys.argv) > 5:
    raise ValueError(usage)

treefile = sys.argv[1]
outbase = ".".join(treefile.split(".")[:-1])

time_ago = 0
format = "png"
max_dimension = 8
if len(sys.argv) > 2:
    time_ago = float(sys.argv[2])
if len(sys.argv) > 3:
    format = sys.argv[3]
if len(sys.argv) > 4:
    max_dimension = float(sys.argv[4])

ts = tskit.load(treefile)

try:
    num_gens = ts.metadata['SLiM']['tick']
except KeyError:
    num_gens = ts.metadata['SLiM']['generation']
params = ts.metadata['SLiM']['user_metadata']
try:
    dt = params['DT'][0]
except KeyError:
    dt = 1.0

if 'WIDTH' in params:
    width = params['WIDTH'][0]
    height = params['HEIGHT'][0]
    xlim = [0.0, width]
    ylim = [0.0, height]
else:
    locs = ts.individual_locations
    xlim = [min(locs[:,0]), max(locs[:,0])]
    width = xlim[1] - xlim[0]
    ylim = [min(locs[:,1]), max(locs[:,1])]
    height = ylim[1] - ylim[0]

size = (max_dimension * max(1.0, width/height), max_dimension * max(1.0, height/width))

fig, ax = plt.subplots(figsize=size)
ax.set_xlabel("eastings")
ax.set_ylabel("northings")
ax.set_title(f"density at {time_ago} ago")

sps.plot_density(ts, time_ago, ax, xlims=xlim, ylims=ylim)

plt.tight_layout()
fig.savefig(outbase + f".density.{time_ago}.{format}", bbox_inches='tight')
