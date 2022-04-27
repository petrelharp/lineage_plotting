#!/usr/bin/env python3

import sys
import pyslim, tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

usage = """
Makes a plot of the current location of individuals.

Usage:
    {} (script name)
""".format(sys.argv[0])

if len(sys.argv) != 2:
    raise ValueError(usage)

treefile = sys.argv[1]
outbase = ".".join(treefile.split(".")[:-1])

ts = pyslim.load(treefile)

num_gens = ts.metadata['SLiM']['generation']
params = ts.metadata['SLiM']['user_metadata']
try:
    dt = params['DT'][0]
except KeyError:
    dt = 1.0
width = params['WIDTH'][0]
height = params['HEIGHT'][0]
size = (8 * max(1.0, width/height), 8 * max(1.0, height/width))

today = ts.individuals_alive_at(0)
locs = np.array([ts.individual(i).location[:2] for i in today])

fig, ax = plt.subplots(figsize=size)
ax.set_xlabel("eastings")
ax.set_ylabel("northings")

sps.plot_density(ts, 0.0, ax, xlims=(0.0, width), ylims=(0.0, height))

fig.savefig(outbase + ".locations.pdf")
