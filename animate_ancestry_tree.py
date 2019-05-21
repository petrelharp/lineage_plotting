#!/usr/bin/env python3

import sys
import pyslim, msprime
import numpy as np
import spatial_slim as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.collections as cs

usage = """
Usage:
    {} (num gens) (num trees) (script name) [KEY=VALUE [KEY=VALUE]]
where the KEY=VALUE pairs get passed to SLiM.
""".format(sys.argv[0])

if len(sys.argv) < 4:
    raise ValueError(usage)

try:
    num_gens = int(sys.argv[1])
except ValueError:
    raise ValueError("First argument should be numeric (a number of generations).")
try:
    num_trees = int(sys.argv[2])
except ValueError:
    raise ValueError("Second argument should be numeric (a number of trees to plot).")
script = sys.argv[3]
kwargs = {}
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    kwargs[k] = v

if 'seed' not in kwargs:
    kwargs['seed'] = np.random.randint(10000)

def animate_tree(ts, children, num_gens):
    """
    An animation of the tree ancestral to an individual.
    """
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    locs = ts.individual_locations
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    # colors
    colormap = lambda x: plt.get_cmap("cool")(x/max(ts.individual_ages))
    # treecolors = [plt.get_cmap("viridis")(x) for x in np.linspace(0, 1, len(children))]
    inds = ts.individuals_by_time(0)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    paths = []
    lc = cs.LineCollection(paths, linewidths=0.5)
    ax.add_collection(lc)

    def update(frame):
        nonlocal children
        nonlocal paths
        inds = ts.individuals_by_time(frame)
        circles.set_offsets(locs[inds,:2])
        # color based on age so far
        circles.set_color(colormap(ts.individuals_age(frame)[inds]))
        newborns = children[ts.individuals_age(frame)[children] == 0]
        pcs = ts.individual_parents(newborns)
        if len(pcs) > 0:
            children = np.concatenate((children, pcs[:,0]))
            paths = paths + [locs[pc,:2] for pc in pcs]
            lc.set_paths(paths)
        return circles, lc

    animation = ani.FuncAnimation(fig, update, 
                                  frames=np.linspace(0, num_gens, num_gens + 1))
    return animation

treefile = sps.run_slim(script = script, **kwargs)
outbase = ".".join(treefile.split(".")[:-1])

ts = sps.SpatialSlimTreeSequence(pyslim.load(treefile), dim=2)

today = np.where(ts.individual_times == 0)[0]
animation = animate_tree(ts, np.random.choice(today, num_trees), num_gens)
animation.save(outbase + ".trees.mp4", writer='ffmpeg')

