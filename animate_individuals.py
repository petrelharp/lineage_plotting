#!/usr/bin/env python3

import sys
import pyslim, tskit
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.collections as cs

usage = """
Usage:
    {} (num gens) (script name) [KEY=VALUE [KEY=VALUE]]
where the KEY=VALUE pairs get passed to SLiM.
""".format(sys.argv[0])

if len(sys.argv) < 3:
    raise ValueError(usage)

try:
    num_gens = int(sys.argv[1])
except ValueError:
    raise ValueError("First argument should be numeric (a number of generations).")
script = sys.argv[2]
kwargs = {}
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    kwargs[k] = v

if 'seed' not in kwargs:
    kwargs['seed'] = np.random.randint(10000)

def animate_individuals(ts, num_gens):
    # an animation of the individuals
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    locs = ts.individual_locations
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    # colors
    colormap = lambda x: plt.get_cmap("cool")(x/max(ts.individual_ages))
    inds = ts.individuals_alive_at(num_gens)
    next_inds = ts.individuals_alive_at(num_gens - 1)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    filled = ax.scatter(locs[next_inds, 0], locs[next_inds, 1], s=10, 
                        facecolors=colormap([0 for _ in next_inds]),
                        edgecolors='none')
    lc = cs.LineCollection([], colors='black', linewidths=0.5)
    ax.add_collection(lc)

    def update(frame):
        inds = ts.individuals_alive_at(frame)
        next_inds = ts.individuals_alive_at(frame - 1)
        circles.set_offsets(locs[inds,:2])
        filled.set_offsets(locs[next_inds,:2])
        # color based on age so far
        circles.set_color(colormap(ts.individual_ages_at(frame)[inds]))
        filled.set_color(colormap(ts.individual_ages_at(frame)[next_inds]))
        if frame > 0:
            new_inds = inds[ts.individual_ages_at(frame)[inds] == 0]
            pcs = ts.individual_parents(new_inds, time=frame)
            lc.set_paths([locs[pc,:2] for pc in pcs])
        return circles, filled, lc

    animation = ani.FuncAnimation(fig, update, 
                                  frames=np.linspace(num_gens, 1, num_gens))
    return animation

treefile = sps.run_slim(script = script, **kwargs)
outbase = ".".join(treefile.split(".")[:-1])

ts = sps.SpatialSlimTreeSequence(pyslim.load(treefile), dim=2)

animation = animate_individuals(ts, num_gens)
animation.save(outbase + ".pop.mp4", writer='ffmpeg')

