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
Makes an animation of the "pedigree trees", i.e., all the ancestors of the
given number of randomly chosen individuals.
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
    locs = ts.individuals_location.reshape((ts.num_individuals,3))
    xmax = np.ceil(max(locs[:,0]))
    ymax = np.ceil(max(locs[:,1]))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    # colors
    colormap = lambda x: plt.get_cmap("cool")(x/max(pyslim.individual_ages(ts)))
    # treecolors = [plt.get_cmap("viridis")(x) for x in np.linspace(0, 1, len(children))]
    inds = pyslim.individuals_alive_at(ts, 0)
    circles = ax.scatter(locs[inds, 0], locs[inds, 1], s=10, 
                         edgecolors=colormap([0 for _ in inds]),
                         facecolors='none')
    paths = []
    lc = cs.LineCollection(paths, linewidths=0.5)
    ax.add_collection(lc)

    def update(frame):
        nonlocal children
        nonlocal paths
        inds = pyslim.individuals_alive_at(ts, frame)
        circles.set_offsets(locs[inds,:2])
        # color based on age so far
        circles.set_color(colormap(pyslim.individual_ages_at(ts, frame)[inds]))
        newborns = children[pyslim.individual_ages_at(ts, frame)[children] == 0]
        pcs = pyslim.individual_parents(ts, newborns)
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

ts = sps.SpatialSlimTreeSequence(tskit.load(treefile), dim=2)

today = np.where(ts.individuals_time == 0)[0]
animation = animate_tree(ts, np.random.choice(today, num_trees), num_gens)
animation.save(outbase + ".trees.mp4", writer='ffmpeg')

